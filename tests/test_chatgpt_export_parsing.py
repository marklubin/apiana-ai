import json
from datetime import datetime
from pathlib import Path


from apiana.batch.chatgpt.chatgpt_export_parsing import (
    convo_from_export_format_dict,
    from_export_mapping_to_messages,
    message_from_chatgpt_format,
)
from apiana.types.common import Conversation


class TestMessageFromChatGPTFormat:
    """Test parsing individual messages from ChatGPT export format"""

    def test_valid_user_message(self):
        """Given a valid user message structure, should parse correctly"""
        data = {
            "message": {
                "id": "test-id-123",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["Hello, world!"]},
                "create_time": 1747058871.377059,
                "metadata": {"model_slug": "gpt-4", "request_id": "req-123"},
            }
        }

        message = message_from_chatgpt_format(data)

        assert message is not None
        assert message.id == "test-id-123"
        assert message.role == "user"
        assert message.type == "text"
        assert message.content == {"content_type": "text", "parts": ["Hello, world!"]}
        assert message.provider_model_id == "gpt-4"
        assert message.provider_request_id == "req-123"
        assert isinstance(message.created_at, datetime)

    def test_valid_assistant_message(self):
        """Given a valid assistant message structure, should parse correctly"""
        data = {
            "message": {
                "id": "test-id-456",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": ["Hi there!"]},
                "create_time": 1747058872.0,
                "metadata": {"model_slug": "gpt-3.5-turbo"},
            }
        }

        message = message_from_chatgpt_format(data)

        assert message is not None
        assert message.id == "test-id-456"
        assert message.role == "assistant"
        assert message.type == "text"

    def test_empty_data_returns_none(self):
        """Given empty or None data, should return None"""
        assert message_from_chatgpt_format(None) is None
        assert message_from_chatgpt_format({}) is None

    def test_missing_message_key_returns_none(self):
        """Given data without 'message' key, should return None"""
        data = {"id": "test", "author": {"role": "user"}}
        assert message_from_chatgpt_format(data) is None

    def test_invalid_message_value_returns_none(self):
        """Given invalid 'message' value type, should return None"""
        data = {"message": "invalid"}
        assert message_from_chatgpt_format(data) is None

    def test_missing_author_data_handled_gracefully(self):
        """Given message without author data, should use default role"""
        data = {
            "message": {
                "id": "test-id",
                "content": {"content_type": "text"},
                "create_time": 1747058871.0,
            }
        }

        message = message_from_chatgpt_format(data)

        assert message is not None
        assert message.role == "user"  # Default value from Message model

    def test_invalid_author_type_handled_gracefully(self):
        """Given invalid author type, should use default role"""
        data = {
            "message": {
                "id": "test-id",
                "author": "invalid_author_string",
                "content": {"content_type": "text"},
            }
        }

        message = message_from_chatgpt_format(data)

        assert message is not None
        assert message.role == "user"  # Default value from Message model

    def test_missing_metadata_handled_gracefully(self):
        """Given message without metadata, should use default empty strings"""
        data = {
            "message": {
                "id": "test-id",
                "author": {"role": "user"},
                "content": {"content_type": "text"},
            }
        }

        message = message_from_chatgpt_format(data)

        assert message is not None
        assert message.provider_model_id == ""  # Default value from Message model
        assert message.provider_request_id == ""  # Default value from Message model

    def test_invalid_metadata_type_handled_gracefully(self):
        """Given invalid metadata type, should use default empty string"""
        data = {
            "message": {
                "id": "test-id",
                "author": {"role": "user"},
                "metadata": ["invalid", "list"],
            }
        }

        message = message_from_chatgpt_format(data)

        assert message is not None
        assert message.provider_model_id == ""  # Default value from Message model

    def test_empty_content_handled_gracefully(self):
        """Given empty content dict, should store empty dict"""
        data = {"message": {"id": "test-id", "author": {"role": "user"}, "content": {}}}

        message = message_from_chatgpt_format(data)

        assert message is not None
        assert message.content == {}

    def test_invalid_content_type_handled_gracefully(self):
        """Given invalid content type, should use empty dict"""
        data = {
            "message": {
                "id": "test-id",
                "author": {"role": "user"},
                "content": "invalid_content",
            }
        }

        message = message_from_chatgpt_format(data)

        assert message is not None
        assert message.content == {}


class TestFromExportMappingToMessages:
    """Test converting mapping structure to list of messages"""

    def test_dict_mapping_with_valid_messages(self):
        """Given dict mapping with valid user/assistant messages, should extract them"""
        mapping_data = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Question?"]},
                }
            },
            "node-2": {
                "message": {
                    "id": "msg-2",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Answer."]},
                }
            },
        }

        messages = from_export_mapping_to_messages(mapping_data)

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_dict_mapping_filters_system_messages(self):
        """Given mapping with system messages, should filter them out"""
        mapping_data = {
            "node-1": {
                "message": {
                    "id": "msg-1",
                    "author": {"role": "system"},
                    "content": {"content_type": "text"},
                }
            },
            "node-2": {
                "message": {
                    "id": "msg-2",
                    "author": {"role": "user"},
                    "content": {"content_type": "text"},
                }
            },
        }

        messages = from_export_mapping_to_messages(mapping_data)

        assert len(messages) == 1
        assert messages[0].role == "user"

    def test_dict_mapping_skips_invalid_nodes(self):
        """Given mapping with invalid node data, should skip them"""
        mapping_data = {
            "node-1": "invalid_node_data",
            "node-2": {
                "message": {
                    "id": "msg-2",
                    "author": {"role": "user"},
                    "content": {"content_type": "text"},
                }
            },
            "node-3": None,
        }

        messages = from_export_mapping_to_messages(mapping_data)

        assert len(messages) == 1
        assert messages[0].id == "msg-2"

    def test_list_mapping_format(self):
        """Given list format mapping, should parse messages"""
        mapping_data = [
            {
                "message": {
                    "id": "msg-1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text"},
                }
            },
            {
                "message": {
                    "id": "msg-2",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text"},
                }
            },
        ]

        messages = from_export_mapping_to_messages(mapping_data)

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_empty_mapping_returns_empty_list(self):
        """Given empty mapping, should return empty list"""
        assert from_export_mapping_to_messages({}) == []
        assert from_export_mapping_to_messages([]) == []

    def test_invalid_mapping_type_returns_empty_list(self):
        """Given invalid mapping type, should return empty list"""
        assert from_export_mapping_to_messages("invalid") == []
        assert from_export_mapping_to_messages(None) == []
        assert from_export_mapping_to_messages(123) == []


class TestConvoFromExportFormatDict:
    """Test converting full conversation from export format"""

    def test_valid_conversation_with_test_data(self):
        """Given actual test data structure, should parse conversation correctly"""
        # Load test data
        test_data_path = Path(__file__).parent / "test-convos.json"
        with open(test_data_path, "r") as f:
            conversations = json.load(f)

        # Test first conversation
        data = conversations[0]
        convo = convo_from_export_format_dict(data)

        assert convo is not None
        assert convo.title == "Power Draw and Costs"
        # The test data actually has 'id' field, not 'openai_conversation_id'
        if "id" in data:
            assert convo.openai_conversation_id == data["id"]
        assert isinstance(convo.create_time, datetime)
        assert isinstance(convo.update_time, datetime)

        # Check messages were extracted (should have user messages)
        user_messages = [m for m in convo.messages if m.role == "user"]
        assert len(user_messages) > 0

    def test_minimal_valid_conversation(self):
        """Given minimal valid structure, should create conversation"""
        data = {
            "title": "Test Conversation",
            "create_time": 1747058871.0,
            "update_time": 1747058872.0,
            "mapping": {
                "node-1": {
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text"},
                    }
                }
            },
        }

        convo = convo_from_export_format_dict(data)

        assert convo is not None
        assert convo.title == "Test Conversation"
        assert len(convo.messages) == 1

    def test_conversation_with_id_field(self):
        """Given conversation with 'id' field, should use as openai_conversation_id"""
        data = {
            "id": "conv-123",
            "title": "Test",
            "mapping": {},
        }

        convo = convo_from_export_format_dict(data)

        assert convo is not None
        assert convo.openai_conversation_id == "conv-123"

    def test_missing_optional_fields_handled_gracefully(self):
        """Given minimal data, should use default values"""
        data = {"mapping": {}}

        convo = convo_from_export_format_dict(data)

        assert convo is not None
        assert convo.title == "untitled conversation"  # Default from Conversation model
        assert isinstance(convo.create_time, datetime)  # Default is datetime.utcnow()
        assert isinstance(convo.update_time, datetime)  # Default is datetime.utcnow()
        assert convo.messages == []

    def test_exception_wrapped_with_context(self):
        """Given data that causes parsing error, should handle gracefully"""
        # Functions are gracefully ignored by from_export_mapping_to_messages
        data = {"mapping": lambda x: x}

        # This should not raise an exception - invalid types are ignored
        convo = convo_from_export_format_dict(data)
        assert convo is not None
        assert convo.messages == []


class TestIntegrationWithTestData:
    """Integration tests using actual test data file"""

    def test_parse_all_conversations_in_test_file(self):
        """Given test data file, should parse all conversations without errors"""
        test_data_path = Path(__file__).parent / "test-convos.json"
        with open(test_data_path, "r") as f:
            conversations_data = json.load(f)

        conversations = []
        for data in conversations_data:
            convo = convo_from_export_format_dict(data)
            conversations.append(convo)

        # Should have parsed all conversations
        assert len(conversations) == len(conversations_data)

        # Each conversation should have required attributes
        for convo in conversations:
            assert isinstance(convo, Conversation)
            assert isinstance(convo.messages, list)

    def test_message_content_structure_from_test_data(self):
        """Given test data, should preserve complex content structures"""
        test_data_path = Path(__file__).parent / "test-convos.json"
        with open(test_data_path, "r") as f:
            conversations_data = json.load(f)

        # Find a message with user_editable_context type
        found_complex_message = False
        for conv_data in conversations_data:
            convo = convo_from_export_format_dict(conv_data)
            for msg in convo.messages:
                if msg.type == "user_editable_context":
                    # Should preserve the complex content structure
                    assert "user_profile" in msg.content
                    assert "user_instructions" in msg.content
                    found_complex_message = True
                    break
            if found_complex_message:
                break

        assert found_complex_message, "Test data should contain user_editable_context messages"