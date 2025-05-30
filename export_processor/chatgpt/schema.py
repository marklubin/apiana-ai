from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List

@dataclass
class Message:
    id: Optional[str] = None
    role: Optional[str] = None
    type: Optional[str] = None
    content: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    provider_model_id: Optional[str] = None
    provider_request_id: Optional[str] = None
    
    @staticmethod
    def parse_timestamp(value) -> Optional[datetime]:
        """Parse timestamp from various formats"""
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value)
            elif isinstance(value, str):
                return datetime.fromisoformat(value)
            elif isinstance(value, datetime):
                return value
        except (ValueError, TypeError, OSError):
            pass
        return None
    
    @classmethod
    def from_chatgpt_format(cls, data: Dict) -> Optional['Message']:
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
        
        try:
            return cls(
                id=message_data.get("id"),
                role=author.get("role"),
                type=content.get("content_type"),
                content=content if content else None,
                created_at=cls.parse_timestamp(message_data.get("create_time")),
                provider_model_id=metadata.get("model_slug"),
                provider_request_id=metadata.get("request_id")
            )
        except Exception:
            return None

@dataclass
class OpenAIConversation:
    title: str = "Untitled Conversation"
    messages: List[Message] = field(default_factory=list)
    openai_conversation_id: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    
    @staticmethod
    def parse_timestamp(value) -> Optional[datetime]:
        """Parse timestamp from various formats"""
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value)
            elif isinstance(value, str):
                return datetime.fromisoformat(value)
            elif isinstance(value, datetime):
                return value
        except (ValueError, TypeError, OSError):
            pass
        return None
    
    @staticmethod
    def parse_messages(mapping_data) -> List[Message]:
        """Extract messages from the mapping structure"""
        if not mapping_data:
            return []
            
        messages = []
        
        try:
            if isinstance(mapping_data, dict):
                for node_data in mapping_data.values():
                    if not isinstance(node_data, dict):
                        continue
                        
                    message = Message.from_chatgpt_format(node_data)
                    if message and message.role in ["user", "assistant"]:
                        messages.append(message)
                        
            elif isinstance(mapping_data, list):
                # Handle if messages are already in list format
                for item in mapping_data:
                    if isinstance(item, dict):
                        message = Message.from_chatgpt_format(item)
                        if message and message.role in ["user", "assistant"]:
                            messages.append(message)
        except Exception:
            # If anything goes wrong, return empty list
            pass
            
        return messages

    @classmethod
    def from_dict(cls, data: Dict) -> Optional['OpenAIConversation']:
        """Create conversation from ChatGPT export JSON"""
        if not data or not isinstance(data, dict):
            return None
            
        try:
            messages = cls.parse_messages(data.get("mapping", {}))
            
            # Skip conversations with no valid messages
            if not messages:
                return None
                
            return cls(
                title=data.get("title", "Untitled Conversation"),
                messages=messages,
                openai_conversation_id=data.get("id"),
                create_time=cls.parse_timestamp(data.get("create_time")),
                update_time=cls.parse_timestamp(data.get("update_time"))
            )
            
        except Exception:
            return None