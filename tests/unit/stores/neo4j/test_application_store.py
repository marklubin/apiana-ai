"""
Unit tests for ApplicationStore with mocked dependencies.

These tests verify the application-level ChatFragment storage functionality.
"""

from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

import pytest

from apiana.stores.neo4j.application_store import ApplicationStore, ChatFragmentNode, PipelineRunNode
from apiana.types.chat_fragment import ChatFragment
from apiana.types.configuration import Neo4jConfig


@pytest.fixture
def mock_neo4j_config() -> Neo4jConfig:
    """Create a mock Neo4j configuration."""
    return Neo4jConfig(
        username="test_user",
        password="test_pass",
        host="localhost",
        port=7687,
        database="test_db",
    )


@pytest.fixture
def sample_fragment() -> ChatFragment:
    """Create a sample chat fragment for testing."""
    return ChatFragment(
        title="Test Conversation",
        messages=[
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        fragment_id="frag-123",
        openai_conversation_id="conv-456",
        create_time=datetime(2024, 1, 15, 10, 0, 0),
        update_time=datetime(2024, 1, 15, 10, 5, 0),
    )


class TestApplicationStore:
    """Unit tests for ApplicationStore."""

    @patch("apiana.stores.neo4j.application_store.db")
    @patch("apiana.stores.neo4j.application_store.neomodel_config")
    def test_init_sets_default_database_url(
        self, mock_neomodel_config, mock_db, mock_neo4j_config
    ):
        """Should set database URL to default database (no suffix)."""
        # Act
        store = ApplicationStore(mock_neo4j_config)

        # Assert
        expected_url = "bolt://test_user:test_pass@localhost:7687"
        assert mock_neomodel_config.DATABASE_URL == expected_url
        mock_db.install_all_labels.assert_called_once()

    @patch("apiana.stores.neo4j.application_store.ChatFragmentNode")
    def test_store_fragment_creates_new_node(
        self, mock_fragment_node_class, mock_neo4j_config, sample_fragment
    ):
        """Should create new ChatFragmentNode when fragment doesn't exist."""
        # Arrange
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = None  # No existing fragment
        mock_fragment_node_class.nodes = mock_nodes

        mock_new_node = Mock()
        mock_fragment_node_class.return_value.save.return_value = mock_new_node

        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Act
            result = store.store_fragment(sample_fragment, tags=["test", "conversation"])

            # Assert
            mock_fragment_node_class.assert_called_once_with(
                fragment_id="frag-123",
                openai_conversation_id="conv-456",
                title="Test Conversation",
                create_time=sample_fragment.create_time,
                update_time=sample_fragment.update_time,
                messages=sample_fragment.messages,
                message_metadata=[],
                tags=["test", "conversation"],
                summary_agent_id=None,
                summary_hash=None
            )

            assert result == mock_new_node

    @patch("apiana.stores.neo4j.application_store.ChatFragmentNode")
    @patch("apiana.stores.neo4j.application_store.datetime")
    def test_store_fragment_updates_existing_node(
        self, mock_datetime, mock_fragment_node_class, mock_neo4j_config, sample_fragment
    ):
        """Should update existing ChatFragmentNode when fragment exists."""
        # Arrange
        test_time = datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.utcnow.return_value = test_time

        mock_existing_node = Mock()
        mock_existing_node.tags = ["old_tag"]
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = mock_existing_node
        mock_fragment_node_class.nodes = mock_nodes

        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Act
            result = store.store_fragment(sample_fragment, tags=["new_tag"])

            # Assert
            assert mock_existing_node.title == "Test Conversation"
            assert mock_existing_node.openai_conversation_id == "conv-456"
            assert mock_existing_node.updated_at == test_time
            assert set(mock_existing_node.tags) == {"old_tag", "new_tag"}
            mock_existing_node.save.assert_called_once()
            assert result == mock_existing_node

    @patch("apiana.stores.neo4j.application_store.ChatFragmentNode")
    def test_get_fragment_found(
        self, mock_fragment_node_class, mock_neo4j_config
    ):
        """Should return ChatFragment when found."""
        # Arrange
        mock_node = Mock()
        mock_node.fragment_id = "frag-123"
        mock_node.openai_conversation_id = "conv-456"
        mock_node.title = "Test"
        mock_node.create_time = datetime(2024, 1, 15, 10, 0, 0)
        mock_node.update_time = datetime(2024, 1, 15, 10, 5, 0)
        mock_node.messages = [{"role": "user", "content": "Hello"}]
        mock_node.message_metadata = []

        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = mock_node
        mock_fragment_node_class.nodes = mock_nodes

        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Act
            result = store.get_fragment("frag-123")

            # Assert
            assert result is not None
            assert result.fragment_id == "frag-123"
            assert result.title == "Test"
            assert len(result.messages) == 1

    @patch("apiana.stores.neo4j.application_store.ChatFragmentNode")
    def test_get_fragment_not_found(
        self, mock_fragment_node_class, mock_neo4j_config
    ):
        """Should return None when fragment not found."""
        # Arrange
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = None
        mock_fragment_node_class.nodes = mock_nodes

        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Act
            result = store.get_fragment("nonexistent")

            # Assert
            assert result is None

    @patch("apiana.stores.neo4j.application_store.ChatFragmentNode")
    def test_list_fragments_with_filters(
        self, mock_fragment_node_class, mock_neo4j_config
    ):
        """Should apply filters when listing fragments."""
        # Arrange
        mock_node1 = Mock()
        mock_node2 = Mock()
        mock_nodes = [mock_node1, mock_node2]
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.__getitem__.return_value = mock_nodes  # Simulate slicing

        mock_fragment_node_class.nodes = mock_query

        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Act
            with patch.object(store, '_node_to_fragment', return_value=Mock()) as mock_convert:
                result = store.list_fragments(
                    limit=10,
                    offset=5,
                    tags=["important"],
                    conversation_id="conv-123"
                )

            # Assert
            mock_query.filter.assert_any_call(openai_conversation_id="conv-123")
            mock_query.filter.assert_any_call(tags__contains="important")
            mock_query.order_by.assert_called_with('-create_time')
            assert len(result) == 2

    @patch("apiana.stores.neo4j.application_store.ChatFragmentNode")
    @patch("apiana.stores.neo4j.application_store.db")
    def test_search_fragments_by_title(
        self, mock_db, mock_fragment_node_class, mock_neo4j_config
    ):
        """Should search fragments by title content."""
        # Arrange
        mock_node = Mock()
        mock_fragment_node_class.inflate.return_value = mock_node
        
        # Mock cypher query results
        mock_db.cypher_query.return_value = ([[mock_node]], None)

        with patch("apiana.stores.neo4j.application_store.neomodel_config"):
            store = ApplicationStore(mock_neo4j_config)

            # Act
            with patch.object(store, '_node_to_fragment', return_value=Mock()) as mock_convert:
                result = store.search_fragments("test query")

            # Assert
            expected_query = """
        MATCH (f:ChatFragmentNode)
        WHERE f.title CONTAINS $search_text
        RETURN f
        LIMIT $limit
        """
            mock_db.cypher_query.assert_called_with(
                expected_query, 
                {"search_text": "test query", "limit": 50}
            )
            assert len(result) == 1

    @patch("apiana.stores.neo4j.application_store.ChatFragmentNode")
    def test_delete_fragment_success(
        self, mock_fragment_node_class, mock_neo4j_config
    ):
        """Should delete fragment when found."""
        # Arrange
        mock_node = Mock()
        # Mock embeddings relationship
        mock_embeddings = Mock()
        mock_embeddings.all.return_value = []  # No embeddings to delete
        mock_node.embeddings = mock_embeddings
        
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = mock_node
        mock_fragment_node_class.nodes = mock_nodes

        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Act
            result = store.delete_fragment("frag-123")

            # Assert
            mock_node.delete.assert_called_once()
            assert result is True

    @patch("apiana.stores.neo4j.application_store.ChatFragmentNode")
    def test_delete_fragment_not_found(
        self, mock_fragment_node_class, mock_neo4j_config
    ):
        """Should return False when fragment not found."""
        # Arrange
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = None
        mock_fragment_node_class.nodes = mock_nodes

        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Act
            result = store.delete_fragment("nonexistent")

            # Assert
            assert result is False

    @patch("apiana.stores.neo4j.application_store.PipelineRunNode")
    def test_create_pipeline_run(
        self, mock_run_node_class, mock_neo4j_config
    ):
        """Should create new PipelineRunNode."""
        # Arrange
        mock_run = Mock()
        mock_run_node_class.return_value.save.return_value = mock_run

        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Act
            result = store.create_pipeline_run(
                "run-123",
                "Test Run",
                config={"param": "value"}
            )

            # Assert
            mock_run_node_class.assert_called_once_with(
                run_id="run-123",
                name="Test Run",
                config={"param": "value"}
            )
            assert result == mock_run

    @patch("apiana.stores.neo4j.application_store.PipelineRunNode")
    @patch("apiana.stores.neo4j.application_store.datetime")
    def test_complete_pipeline_run(
        self, mock_datetime, mock_run_node_class, mock_neo4j_config
    ):
        """Should complete processor run with stats."""
        # Arrange
        test_time = datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.utcnow.return_value = test_time

        mock_run = Mock()
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = mock_run
        mock_run_node_class.nodes = mock_nodes

        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Act
            result = store.complete_pipeline_run(
                "run-123",
                stats={"processed": 10},
                errors=["error1"]
            )

            # Assert
            assert mock_run.completed_at == test_time
            assert mock_run.status == "failed"  # Because errors were provided
            assert mock_run.stats == {"processed": 10}
            assert mock_run.errors == ["error1"]
            mock_run.save.assert_called_once()
            assert result == mock_run

    @patch("apiana.stores.neo4j.application_store.PipelineRunNode")
    @patch("apiana.stores.neo4j.application_store.ChatFragmentNode")
    def test_link_fragments_to_run(
        self, mock_fragment_node_class, mock_run_node_class, mock_neo4j_config
    ):
        """Should link fragments to processor run."""
        # Arrange
        mock_run = Mock()
        mock_run.processed_fragments = Mock()
        mock_run_nodes = Mock()
        mock_run_nodes.filter.return_value.first.return_value = mock_run
        mock_run_node_class.nodes = mock_run_nodes

        mock_fragment = Mock()
        mock_fragment.processed_by_runs = ["old-run"]
        mock_fragment_nodes = Mock()
        mock_fragment_nodes.filter.return_value.first.return_value = mock_fragment
        mock_fragment_node_class.nodes = mock_fragment_nodes

        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Act
            result = store.link_fragments_to_run("run-123", ["frag-1"])

            # Assert
            mock_run.processed_fragments.connect.assert_called_once_with(mock_fragment)
            assert "run-123" in mock_fragment.processed_by_runs
            mock_fragment.save.assert_called_once()
            assert result == 1

    @patch("apiana.stores.neo4j.application_store.PipelineRunNode")
    def test_get_pipeline_runs(
        self, mock_run_node_class, mock_neo4j_config
    ):
        """Should return recent processor runs."""
        # Arrange
        mock_runs = [Mock(), Mock()]
        mock_slice_result = MagicMock()
        mock_slice_result.__getitem__.return_value = mock_runs
        
        mock_order_result = MagicMock()
        mock_order_result.__getitem__.return_value = mock_runs
        
        mock_nodes = Mock()
        mock_nodes.order_by.return_value = mock_order_result
        mock_run_node_class.nodes = mock_nodes

        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Act
            result = store.get_pipeline_runs(limit=10)

            # Assert
            mock_nodes.order_by.assert_called_with('-started_at')
            assert result == mock_runs

    @patch("apiana.stores.neo4j.application_store.EmbeddingNode")
    @patch("apiana.stores.neo4j.application_store.PipelineRunNode")
    @patch("apiana.stores.neo4j.application_store.ChatFragmentNode")
    @patch("apiana.stores.neo4j.application_store.datetime")
    def test_get_run_statistics(
        self, mock_datetime, mock_fragment_node_class, mock_run_node_class, 
        mock_embedding_node_class, mock_neo4j_config
    ):
        """Should return statistics about stored data."""
        # Arrange
        mock_datetime.utcnow.return_value.replace.return_value = datetime(2024, 1, 15, 0, 0, 0)

        # Mock fragment nodes
        mock_fragment_nodes = MagicMock()
        mock_fragment_nodes.__len__.return_value = 100
        mock_fragment_filter_result = MagicMock()
        mock_fragment_filter_result.__len__.return_value = 5
        mock_fragment_nodes.filter.return_value = mock_fragment_filter_result
        mock_fragment_node_class.nodes = mock_fragment_nodes

        # Mock run nodes
        mock_run_nodes = MagicMock()
        mock_run_nodes.__len__.return_value = 20
        mock_run_node_class.nodes = mock_run_nodes
        
        # Mock embedding nodes
        mock_embedding_nodes = MagicMock()
        mock_embedding_nodes.__len__.return_value = 50
        mock_embedding_node_class.nodes = mock_embedding_nodes

        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Act
            result = store.get_run_statistics()

            # Assert
            assert result["total_fragments"] == 100
            assert result["total_pipeline_runs"] == 20
            assert result["total_embeddings"] == 50
            assert result["fragments_added_today"] == 5

    def test_node_to_fragment_conversion(self, mock_neo4j_config):
        """Should correctly convert ChatFragmentNode to ChatFragment."""
        with patch("apiana.stores.neo4j.application_store.neomodel_config"), \
             patch("apiana.stores.neo4j.application_store.db"):
            
            store = ApplicationStore(mock_neo4j_config)

            # Create mock node
            mock_node = Mock()
            mock_node.fragment_id = "frag-123"
            mock_node.openai_conversation_id = "conv-456"
            mock_node.title = "Test"
            mock_node.create_time = datetime(2024, 1, 15, 10, 0, 0)
            mock_node.update_time = datetime(2024, 1, 15, 10, 5, 0)
            mock_node.messages = [{"role": "user", "content": "Hello"}]
            mock_node.message_metadata = [{"timestamp": "2024-01-15"}]

            # Act
            result = store._node_to_fragment(mock_node)

            # Assert
            assert isinstance(result, ChatFragment)
            assert result.fragment_id == "frag-123"
            assert result.openai_conversation_id == "conv-456"
            assert result.title == "Test"
            assert len(result.messages) == 1
            assert len(result.message_metadata) == 1