import json
from datetime import datetime
import pytest
from apiana.types.chatgpt_conversation import Message, ChatGPTConversation


class TestMessageSerialization:
    def test_message_to_dict_basic(self):
        """Test basic Message to_dict conversion"""
        msg = Message(
            id="test-id",
            role="user",
            type="text",
            content={"text": "Hello world"},
            provider_model_id="gpt-4",
        )

        result = msg.to_dict()

        assert result["id"] == "test-id"
        assert result["role"] == "user"
        assert result["type"] == "text"
        assert result["content"] == {"text": "Hello world"}
        assert result["provider_model_id"] == "gpt-4"
        assert result["created_at"] is None
        assert result["provider_request_id"] is None

    def test_message_to_dict_with_datetime(self):
        """Test Message to_dict with datetime conversion"""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        msg = Message(id="test-id", role="assistant", created_at=dt)

        result = msg.to_dict()

        assert result["created_at"] == "2024-01-15T10:30:00"
        assert isinstance(result["created_at"], str)

    def test_direct_json_dumps_message_fails(self):
        """Test that direct json.dumps() on Message fails"""
        msg = Message(id="test-id", role="user", content={"text": "Hello"})

        # This should raise TypeError
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps(msg)

    def test_message_json_serializable(self):
        """Test that Message.to_dict() output is JSON serializable"""
        msg = Message(
            id="test-id",
            role="user",
            type="text",
            content={"text": "Test message", "parts": ["part1", "part2"]},
            created_at=datetime.now(),
            provider_model_id="gpt-4",
            provider_request_id="req-123",
        )

        result = msg.to_dict()

        # Should not raise any exception
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Verify we can parse it back
        parsed = json.loads(json_str)
        assert parsed["id"] == "test-id"
        assert parsed["content"]["text"] == "Test message"


class TestChatGPTConversationSerialization:
    def test_conversation_to_dict_basic(self):
        """Test basic ChatGPTConversation to_dict conversion"""
        conv = ChatGPTConversation(
            title="Test Conversation", openai_conversation_id="conv-123"
        )

        result = conv.to_dict()

        assert result["title"] == "Test Conversation"
        assert result["messages"] == []
        assert result["openai_conversation_id"] == "conv-123"
        assert result["create_time"] is None
        assert result["update_time"] is None

    def test_conversation_to_dict_with_messages(self):
        """Test ChatGPTConversation to_dict with messages"""
        msg1 = Message(id="1", role="user", content={"text": "Hello"})
        msg2 = Message(id="2", role="assistant", content={"text": "Hi there"})

        conv = ChatGPTConversation(
            title="Chat", messages=[msg1, msg2], openai_conversation_id="conv-123"
        )

        result = conv.to_dict()

        assert len(result["messages"]) == 2
        assert result["messages"][0]["id"] == "1"
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["id"] == "2"
        assert result["messages"][1]["role"] == "assistant"

    def test_conversation_to_dict_with_datetimes(self):
        """Test ChatGPTConversation to_dict with datetime conversion"""
        create_dt = datetime(2024, 1, 1, 12, 0, 0)
        update_dt = datetime(2024, 1, 2, 14, 30, 0)

        conv = ChatGPTConversation(
            title="Test", create_time=create_dt, update_time=update_dt
        )

        result = conv.to_dict()

        assert result["create_time"] == "2024-01-01T12:00:00"
        assert result["update_time"] == "2024-01-02T14:30:00"

    def test_conversation_to_json(self):
        """Test ChatGPTConversation to_json method"""
        msg = Message(
            id="msg-1",
            role="user",
            content={"text": "Test message"},
            created_at=datetime(2024, 1, 15, 10, 0, 0),
        )

        conv = ChatGPTConversation(
            title="Test Conversation",
            messages=[msg],
            openai_conversation_id="conv-123",
            create_time=datetime(2024, 1, 15, 9, 0, 0),
            update_time=datetime(2024, 1, 15, 10, 0, 0),
        )

        json_str = conv.to_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)

        assert parsed["title"] == "Test Conversation"
        assert parsed["openai_conversation_id"] == "conv-123"
        assert len(parsed["messages"]) == 1
        assert parsed["messages"][0]["id"] == "msg-1"
        assert parsed["messages"][0]["created_at"] == "2024-01-15T10:00:00"
        assert parsed["create_time"] == "2024-01-15T09:00:00"
        assert parsed["update_time"] == "2024-01-15T10:00:00"

    def test_conversation_to_json_with_indent(self):
        """Test ChatGPTConversation to_json with indentation"""
        conv = ChatGPTConversation(title="Test")

        json_str = conv.to_json(indent=2)

        # Should contain newlines and spaces for indentation
        assert "\n" in json_str
        assert "  " in json_str

    def test_direct_json_dumps_fails(self):
        """Test that direct json.dumps() on ChatGPTConversation fails"""
        conv = ChatGPTConversation(
            title="Test",
            messages=[Message(id="1", role="user", content={"text": "Hello"})],
        )

        # This should raise TypeError
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps(conv)

    def test_json_dumps_with_to_dict_works(self):
        """Test that json.dumps() works when using to_dict()"""
        conv = ChatGPTConversation(
            title="Test",
            messages=[Message(id="1", role="user", content={"text": "Hello"})],
        )

        # This should work
        json_str = json.dumps(conv.to_dict())
        assert isinstance(json_str, str)

        # Verify content
        parsed = json.loads(json_str)
        assert parsed["title"] == "Test"
        assert len(parsed["messages"]) == 1

    def test_roundtrip_serialization(self):
        """Test that we can serialize and deserialize maintaining data integrity"""
        original_data = {
            "id": "conv-456",
            "title": "Roundtrip Test",
            "create_time": 1705320000.0,  # Unix timestamp
            "update_time": 1705323600.0,
            "mapping": {
                "node1": {
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "text": "Hello"},
                        "create_time": 1705320000.0,
                        "metadata": {"model_slug": "gpt-4"},
                    }
                },
                "node2": {
                    "message": {
                        "id": "msg-2",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "text": "Hi!"},
                        "create_time": 1705320060.0,
                        "metadata": {"model_slug": "gpt-4", "request_id": "req-789"},
                    }
                },
            },
        }

        # Create conversation from dict
        conv = ChatGPTConversation.from_export_format_dict(original_data)
        assert conv is not None

        # Serialize to dict
        serialized = conv.to_dict()

        # Verify key data is preserved
        assert serialized["title"] == "Roundtrip Test"
        assert serialized["openai_conversation_id"] == "conv-456"
        assert len(serialized["messages"]) == 2

        # Verify messages
        assert serialized["messages"][0]["role"] == "user"
        assert serialized["messages"][1]["role"] == "assistant"

        # Verify JSON serialization works
        json_str = conv.to_json()
        parsed = json.loads(json_str)
        assert parsed["title"] == "Roundtrip Test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
