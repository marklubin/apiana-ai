import json
import os
import tempfile

import pytest

from apiana.batch.chatgpt.chatgpt_export_loader import ChatGPTExportLoader


class TestChatGPTExportProcessor:
    @pytest.fixture
    def loader(self):
        return ChatGPTExportLoader()

    def test_load_valid_conversation(self, loader):
        # Given: A valid JSON file with a simple conversation
        valid_data = [
            {
                "id": "conv-123",
                "title": "Test Conversation",
                "create_time": 1747058871.362972,
                "update_time": 1747068677.782826,
                "mapping": {
                    "msg-1": {
                        "id": "msg-1",
                        "message": {
                            "id": "msg-1",
                            "author": {"role": "user"},
                            "create_time": 1747058871.377059,
                            "content": {
                                "content_type": "text",
                                "parts": ["Hello, how are you?"]
                            },
                            "metadata": {}
                        }
                    },
                    "msg-2": {
                        "id": "msg-2",
                        "message": {
                            "id": "msg-2",
                            "author": {"role": "assistant"},
                            "create_time": 1747058871.818458,
                            "content": {
                                "content_type": "text",
                                "parts": ["I'm doing well, thank you!"]
                            },
                            "metadata": {
                                "model_slug": "gpt-4o"
                            }
                        }
                    }
                }
            }
        ]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_data, f)
            temp_path = f.name
        try:
            # When: load() is called with the file path
            conversations = loader.load(temp_path)
            # Then: Returns list with one ChatGPTConversation object
            assert len(conversations) == 1
            conv = conversations[0]
            assert conv.title == "Test Conversation"
            assert conv.openai_conversation_id == "conv-123"
            assert len(conv.messages) == 2
            assert conv.messages[0].role == "user"
            assert conv.messages[0].content["parts"][0] == "Hello, how are you?"
            assert conv.messages[1].role == "assistant"
            assert conv.messages[1].content["parts"][0] == "I'm doing well, thank you!"
            assert conv.messages[1].provider_model_id == "gpt-4o"
        finally:
            os.unlink(temp_path)