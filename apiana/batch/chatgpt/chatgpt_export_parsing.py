import logging
from typing import Dict, List, Optional

from apiana import utils
from apiana.types.common import Conversation, Message

logger = logging.getLogger(__name__)


def message_from_chatgpt_format(data: Dict) -> Optional["Message"]:
    """Create Message from ChatGPT export format"""
    if not data or not isinstance(data, dict):
        return None

    message_data = data.get("message", {})
    if not message_data or not isinstance(message_data, dict):
        return None

    # Safely extract author info
    author = message_data.get("author", {})
    if not isinstance(author, dict):
        author = {}

    # Extract metadata safely
    metadata = message_data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    # Extract content safely
    content = message_data.get("content", {})
    if not isinstance(content, dict):
        content = {}

    # Build kwargs dict with only non-None values
    kwargs = {}
    
    # Set required fields
    if message_data.get("id"):
        kwargs["id"] = message_data["id"]
    
    if author.get("role"):
        kwargs["role"] = author["role"]
    
    if content.get("content_type"):
        kwargs["type"] = content["content_type"]
    
    # Only set content if it's not empty
    if content:
        kwargs["content"] = content
    
    # Parse timestamp if present
    create_time = utils.parse_timestamp(message_data.get("create_time"))
    if create_time:
        kwargs["created_at"] = create_time
    
    # Set provider fields if present
    if metadata.get("model_slug"):
        kwargs["provider_model_id"] = metadata["model_slug"]
    
    if metadata.get("request_id"):
        kwargs["provider_request_id"] = metadata["request_id"]
    
    return Message(**kwargs)


def convo_from_export_format_dict(data: Dict) -> Optional["Conversation"]:
    try:
        messages = from_export_mapping_to_messages(data.get("mapping", {}))

        # Build kwargs dict with only non-None values
        kwargs = {"messages": messages}
        
        if data.get("title"):
            kwargs["title"] = data["title"]
            
        if data.get("id"):
            kwargs["openai_conversation_id"] = data["id"]
            
        create_time = utils.parse_timestamp(data.get("create_time"))
        if create_time:
            kwargs["create_time"] = create_time
            
        update_time = utils.parse_timestamp(data.get("update_time"))
        if update_time:
            kwargs["update_time"] = update_time

        return Conversation(**kwargs)

    except Exception as e:
        raise Exception("Failed to convert ChatGPT export JSON", e)


def from_export_mapping_to_messages(mapping_data) -> List[Message]:
    messages = []

    if isinstance(mapping_data, dict):
        for node_data in mapping_data.values():
            if not isinstance(node_data, dict):
                continue

            message = message_from_chatgpt_format(node_data)
            if message and message.role in ["user", "assistant"]:
                messages.append(message)

    elif isinstance(mapping_data, list):
        # Handle if messages are already in list format
        for item in mapping_data:
            if isinstance(item, dict):
                message = message_from_chatgpt_format(item)
                if message and message.role in ["user", "assistant"]:
                    messages.append(message)

    return messages
