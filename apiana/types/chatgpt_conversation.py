from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict, List
import json

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Message to dictionary with JSON-serializable types"""
        data = asdict(self)
        # Convert datetime to ISO format string
        if data.get('created_at') and isinstance(data['created_at'], datetime):
            data['created_at'] = data['created_at'].isoformat()
        return data

@dataclass
class ChatGPTConversation:
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
    def from_dict(cls, data: Dict) -> Optional['ChatGPTConversation']:
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ChatGPTConversation to dictionary with JSON-serializable types"""
        data = {
            'title': self.title,
            'messages': [msg.to_dict() for msg in self.messages],
            'openai_conversation_id': self.openai_conversation_id,
            'create_time': self.create_time.isoformat() if self.create_time else None,
            'update_time': self.update_time.isoformat() if self.update_time else None
        }
        return data
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert ChatGPTConversation to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)