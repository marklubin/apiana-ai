import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


from apiana.applications.chatgpt_export.cli import (
    embedded_chatgpt_export_summaries,
    get_dependencies,
    process_one_conversation,
    write_convos_in_apiana_format,
)
from apiana.types.common import Conversation, Message


class TestWriteConvosInApianaFormat:
    """Test writing conversations to JSON files"""

    def test_write_single_conversation(self):
        """Given a single conversation, should write to JSON file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test conversation
            convo = Conversation(
                title="Test Chat",
                messages=[
                    Message(id="1", role="user", content={"text": "Hello"}),
                    Message(id="2", role="assistant", content={"text": "Hi"}),
                ],
            )

            # Write conversations
            write_convos_in_apiana_format([convo], temp_dir)

            # Check output
            parsed_dir = Path(temp_dir) / "parsed"
            assert parsed_dir.exists()

            output_file = parsed_dir / "0_test_chat.json"
            assert output_file.exists()

            # Verify content
            with open(output_file) as f:
                data = json.load(f)
                assert data["title"] == "Test Chat"
                assert len(data["messages"]) == 2

    def test_write_multiple_conversations(self):
        """Given multiple conversations, should write each to separate file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            convos = [
                Conversation(title="Chat One", messages=[]),
                Conversation(title="Chat Two", messages=[]),
                Conversation(title="Chat Three", messages=[]),
            ]

            write_convos_in_apiana_format(convos, temp_dir)

            parsed_dir = Path(temp_dir) / "parsed"
            files = list(parsed_dir.glob("*.json"))
            assert len(files) == 3

            # Check file names
            file_names = [f.name for f in files]
            assert "0_chat_one.json" in file_names
            assert "1_chat_two.json" in file_names
            assert "2_chat_three.json" in file_names

    def test_handle_special_characters_in_title(self):
        """Given title with special characters, should sanitize filename"""
        with tempfile.TemporaryDirectory() as temp_dir:
            convo = Conversation(
                title="Chat: With Special/Characters & Symbols!",
                messages=[],
            )

            write_convos_in_apiana_format([convo], temp_dir)

            parsed_dir = Path(temp_dir) / "parsed"
            files = list(parsed_dir.glob("*.json"))
            assert len(files) == 1
            # Special characters are sanitized
            assert "0_chat__with_special_characters__symbols.json" == files[0].name.lower()

    def test_creates_parsed_directory_if_not_exists(self):
        """Given output dir without parsed subdir, should create it"""
        with tempfile.TemporaryDirectory() as temp_dir:
            assert not (Path(temp_dir) / "parsed").exists()

            write_convos_in_apiana_format([], temp_dir)

            assert (Path(temp_dir) / "parsed").exists()


class TestProcessOneConversation:
    """Test processing individual conversations"""

    def test_process_conversation_with_mocked_dependencies(self):
        """Given a conversation and mocked dependencies, should process correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test conversation
            convo = Conversation(
                title="Test Chat",
                messages=[
                    Message(id="1", role="user", content={"text": "Hi"}),
                ],
            )

            # Mock dependencies
            mock_memory_store = MagicMock()
            mock_summarizer = MagicMock()
            mock_summarizer.invoke.return_value = MagicMock(
                content="Short test summary"
            )
            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]  # Mock vector

            # Process conversation
            process_one_conversation(
                convo, temp_dir, mock_memory_store, mock_summarizer, mock_embedder
            )

            # Verify summarizer was called
            mock_summarizer.invoke.assert_called_once()
            prompt_arg = mock_summarizer.invoke.call_args[0][0]
            assert "Test Chat" in prompt_arg  # Title should be in prompt

            # Verify embedder was called with summary
            mock_embedder.embed_query.assert_called_once_with("Short test summary")

            # Verify memory store was called
            mock_memory_store.store_convo.assert_called_once_with(
                convo, "Short test summary", [0.1, 0.2, 0.3], []
            )

            # Verify summary file was written
            summary_file = Path(temp_dir) / "test_chat.txt"
            assert summary_file.exists()
            assert summary_file.read_text() == "Short test summary"

    def test_process_conversation_with_real_embedder(self):
        """Given a conversation with real embedder, should generate actual embeddings"""
        with tempfile.TemporaryDirectory() as temp_dir:
            convo = Conversation(
                title="Quick Test",
                messages=[Message(id="1", role="user", content={"text": "Hello"})],
            )

            # Mock only LLM and memory store
            mock_memory_store = MagicMock()
            mock_summarizer = MagicMock()
            mock_summarizer.invoke.return_value = MagicMock(content="Quick summary")

            # Use real embedder with small model
            from neo4j_graphrag.embeddings.sentence_transformers import (
                SentenceTransformerEmbeddings,
            )

            real_embedder = SentenceTransformerEmbeddings(
                "sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True
            )

            # Process
            process_one_conversation(
                convo, temp_dir, mock_memory_store, mock_summarizer, real_embedder
            )

            # Verify vector was generated
            call_args = mock_memory_store.store_convo.call_args[0]
            vector = call_args[2]
            assert isinstance(vector, list)
            assert len(vector) > 0  # Should have embedding dimensions
            assert all(isinstance(x, float) for x in vector)

    def test_handles_conversation_with_special_characters(self):
        """Given conversation with special chars, should handle filename correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            convo = Conversation(
                title="Test: Special/Chars",
                messages=[],
            )

            mock_deps = (MagicMock(), MagicMock(), MagicMock())
            mock_deps[1].invoke.return_value = MagicMock(content="summary")
            mock_deps[2].embed_query.return_value = [0.1]

            process_one_conversation(convo, temp_dir, *mock_deps)

            # Check that file was created with sanitized name
            files = list(Path(temp_dir).glob("*.txt"))
            assert len(files) == 1
            assert "test__special_chars.txt" == files[0].name.lower()


class TestEmbeddedChatGPTExportSummaries:
    """Test the main processing function"""

    def test_full_pipeline_with_mocked_dependencies(self):
        """Given export file and mocked deps, should process all conversations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test input file
            test_data = [
                {
                    "title": "Chat 1",
                    "mapping": {
                        "node1": {
                            "message": {
                                "id": "1",
                                "author": {"role": "user"},
                                "content": {"content_type": "text"},
                            }
                        }
                    },
                },
                {
                    "title": "Chat 2",
                    "mapping": {},
                },
            ]

            input_file = Path(temp_dir) / "test_export.json"
            with open(input_file, "w") as f:
                json.dump(test_data, f)

            # Mock dependencies
            mock_memory_store = MagicMock()
            mock_summarizer = MagicMock()
            mock_summarizer.invoke.return_value = MagicMock(content="summary")
            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [0.1, 0.2]

            # Run the pipeline
            embedded_chatgpt_export_summaries(
                str(input_file),
                temp_dir,
                mock_memory_store,
                mock_summarizer,
                mock_embedder,
            )

            # Verify parsed files were created
            parsed_dir = Path(temp_dir) / "parsed"
            parsed_files = list(parsed_dir.glob("*.json"))
            assert len(parsed_files) == 2

            # Verify summaries were created
            summaries_dir = Path(temp_dir) / "summaries"
            summary_files = list(summaries_dir.glob("*.txt"))
            assert len(summary_files) == 2

            # Verify memory store was called for each conversation
            assert mock_memory_store.store_convo.call_count == 2

    def test_handles_empty_export_file(self):
        """Given empty export file, should handle gracefully"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "empty.json"
            with open(input_file, "w") as f:
                json.dump([], f)

            mock_deps = (MagicMock(), MagicMock(), MagicMock())

            embedded_chatgpt_export_summaries(str(input_file), temp_dir, *mock_deps)

            # Should create directories but no files
            assert (Path(temp_dir) / "parsed").exists()
            assert (Path(temp_dir) / "summaries").exists()
            assert len(list((Path(temp_dir) / "parsed").glob("*.json"))) == 0

    def test_integration_with_test_data(self):
        """Given actual test data file, should process without errors"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use the actual test data file
            test_data_file = Path(__file__).parent / "test-convos.json"

            if test_data_file.exists():
                # Mock dependencies
                mock_memory_store = MagicMock()
                mock_summarizer = MagicMock()
                mock_summarizer.invoke.return_value = MagicMock(content="test summary")
                mock_embedder = MagicMock()
                mock_embedder.embed_query.return_value = [0.1] * 384  # Mock embedding

                # Run pipeline
                embedded_chatgpt_export_summaries(
                    str(test_data_file),
                    temp_dir,
                    mock_memory_store,
                    mock_summarizer,
                    mock_embedder,
                )

                # Verify files were created
                parsed_files = list((Path(temp_dir) / "parsed").glob("*.json"))
                assert len(parsed_files) > 0

                # Verify first file has expected structure
                with open(parsed_files[0]) as f:
                    data = json.load(f)
                    assert "title" in data
                    assert "messages" in data


class TestGetDependencies:
    """Test dependency initialization"""

    @patch("apiana.applications.chatgpt_export.cli.Neo4jMemoryStore")
    @patch("apiana.applications.chatgpt_export.cli.OpenAILLM")
    @patch("apiana.applications.chatgpt_export.cli.SentenceTransformerEmbeddings")
    def test_initializes_dependencies_with_config(
        self, mock_embedder_class, mock_llm_class, mock_store_class
    ):
        """Given runtime config, should initialize dependencies correctly"""
        from apiana import runtime_config

        # Call get_dependencies
        memory_store, summarizer, embedder = get_dependencies()

        # Verify Neo4j store was initialized with config
        mock_store_class.assert_called_once_with(runtime_config.neo4j)

        # Verify LLM was initialized with config
        mock_llm_class.assert_called_once_with(
            runtime_config.summarizer.model_name,
            {
                "temperature": runtime_config.summarizer.temperature,
                "max_tokens": runtime_config.summarizer.max_tokens,
            },
            base_url=runtime_config.summarizer.inference_provider_config.base_url,
        )

        # Verify embedder was initialized
        mock_embedder_class.assert_called_once_with(
            runtime_config.embedding_model_name, trust_remote_code=True
        )

        # Verify return values
        assert memory_store == mock_store_class.return_value
        assert summarizer == mock_llm_class.return_value
        assert embedder == mock_embedder_class.return_value


class TestConfigurationOverride:
    """Test that unit-test configuration is loaded correctly"""

    def test_unit_test_config_loaded(self):
        """Given APIANA_ENV_STAGE=unit-test, should load test config"""
        # Set the environment variable
        os.environ["APIANA_ENV_STAGE"] = "unit-test"
        
        # Re-import to get fresh config
        import importlib
        import apiana
        importlib.reload(apiana)
        
        from apiana import runtime_config

        # Verify we're using unit-test config
        assert runtime_config.environment_stage == "unit-test"
        assert runtime_config.neo4j.port == 7688  # Test-specific port
        assert runtime_config.neo4j.database == "test"
        assert runtime_config.summarizer.model_name == "mock-model"
        assert runtime_config.summarizer.max_tokens == 50  # Smaller for tests
        assert (
            runtime_config.embedding_model_name
            == "sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_prompt_files_exist_for_test_config(self):
        """Given test config, prompt files should exist"""
        from apiana import runtime_config

        if runtime_config.environment_stage == "unit-test":
            # Check that test prompt files exist
            test_fixtures = Path(__file__).parent / "fixtures"
            assert (test_fixtures / "test-system-prompt.txt").exists()
            assert (test_fixtures / "test-user-prompt.txt").exists()

            # Verify prompt config loaded them
            assert "test assistant" in runtime_config.summarizer.prompt_config.system_prompt
            assert "10 words or less" in runtime_config.summarizer.prompt_config.userprompt_template