"""
Configuration classes for the Apiana processing system.

This module provides configuration management for different environments:
- local: Everything runs on the local machine
- dev: Uses remote Ollama and development Neo4j instances
- production: Uses 3rd party APIs (OpenAI) and hosted services
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import uuid
from datetime import datetime


@dataclass
class PromptConfig:
    """Configuration for prompt templates."""
    name: str
    system_prompt_file: str
    user_prompt_template_file: str


@dataclass
class LLMProviderConfig:
    """Configuration for LLM provider.
    
    All providers use OpenAI-compatible APIs, so no type field is needed.
    """
    model: str = "gpt-4"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class EmbedderConfig:
    """Configuration for embeddings."""
    type: str = "local"  # "local", "openai", etc.
    model: str = "nomic-embed-text"
    dimension: int = 768


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j database connection.
    """
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: Optional[str] = None  # None uses default database
    
    @property
    def auth(self) -> Tuple[str, str]:
        """Return auth tuple for Neo4j driver."""
        return (self.username, self.password)


def json_encoder(obj: Any) -> Any:
    """Custom JSON encoder for dataclasses and datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif dataclass.__class__.__name__ in str(type(obj)):
        return asdict(obj)
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass
class ChatGPTExportProcessorConfiguration:
    """Main configuration for the conversation processor.
    
    This can be loaded from TOML files for different environments.
    """
    # General settings
    environment: str = "local"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    batch_size: int = 10
    output_base_dir: str = "output"
    
    # Component configurations
    prompt: PromptConfig = field(default_factory=lambda: PromptConfig(
        name="default",
        system_prompt_file="self-reflective-system-message.txt",
        user_prompt_template_file="self-reflective-prompt-template.txt"
    ))
    llm_provider: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    
    def get_output_dir(self) -> Path:
        """Get the output directory for this run."""
        return Path(self.output_base_dir) / self.run_id
    
    def get_summaries_dir(self) -> Path:
        """Get the summaries subdirectory for this run."""
        return self.get_output_dir() / "summaries"
    
    def save_to_file(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self, f, default=json_encoder, indent=2)
    
    @classmethod
    def from_file(cls, path: Path) -> 'ChatGPTExportProcessorConfiguration':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            environment=config_dict.get('environment', 'local'),
            run_id=config_dict.get('run_id', str(uuid.uuid4())),
            prompt=PromptConfig(**config_dict.get('prompt', {})),
            llm_provider=LLMProviderConfig(**config_dict.get('llm_provider', {})),
            embedder=EmbedderConfig(**config_dict.get('embedder', {})),
            neo4j=Neo4jConfig(**config_dict.get('neo4j', {})),
            batch_size=config_dict.get('batch_size', 10),
            output_base_dir=config_dict.get('output_base_dir', 'output')
        )
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary using dataclasses.asdict."""
        return asdict(self)
    
