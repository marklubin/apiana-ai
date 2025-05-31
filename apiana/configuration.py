"""
Configuration classes for the Apiana processing system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import uuid


@dataclass
class PromptConfig:
    """Configuration for prompt templates."""
    name: str
    system_prompt_file: str
    user_prompt_template_file: str


@dataclass
class LLMProviderConfig:
    """Configuration for LLM provider."""
    type: str = "openai"  # "openai", "ollama", etc.
    model: str = "gpt-4"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'model': self.model,
            'base_url': self.base_url,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }


@dataclass
class EmbedderConfig:
    """Configuration for embeddings."""
    type: str = "local"  # "local", "openai", etc.
    model: str = "nomic-embed-text"
    dimension: int = 768
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'model': self.model,
            'dimension': self.dimension
        }


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j database connection."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: Optional[str] = None  # None uses default database
    
    @property
    def auth(self) -> Tuple[str, str]:
        """Return auth tuple for Neo4j driver."""
        return (self.username, self.password)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'uri': self.uri,
            'username': self.username,
            'database': self.database
        }


@dataclass
class ProcessorConfig:
    """Main configuration for the conversation processor."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: PromptConfig = field(default_factory=lambda: PromptConfig(
        name="default",
        system_prompt_file="self-reflective-system-message.txt",
        user_prompt_template_file="self-reflective-prompt-template.txt"
    ))
    llm_provider: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    batch_size: int = 10
    output_base_dir: str = "output"
    
    def get_output_dir(self) -> Path:
        """Get the output directory for this run."""
        return Path(self.output_base_dir) / self.run_id
    
    def get_summaries_dir(self) -> Path:
        """Get the summaries subdirectory for this run."""
        return self.get_output_dir() / "summaries"
    
    def save_to_file(self, path: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'run_id': self.run_id,
            'prompt': {
                'name': self.prompt.name,
                'system_prompt_file': self.prompt.system_prompt_file,
                'user_prompt_template_file': self.prompt.user_prompt_template_file
            },
            'llm_provider': self.llm_provider.to_dict(),
            'embedder': self.embedder.to_dict(),
            'neo4j': self.neo4j.to_dict(),
            'batch_size': self.batch_size,
            'output_base_dir': self.output_base_dir
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_file(cls, path: Path) -> 'ProcessorConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            run_id=config_dict.get('run_id', str(uuid.uuid4())),
            prompt=PromptConfig(**config_dict.get('prompt', {})),
            llm_provider=LLMProviderConfig(**config_dict.get('llm_provider', {})),
            embedder=EmbedderConfig(**config_dict.get('embedder', {})),
            neo4j=Neo4jConfig(**config_dict.get('neo4j', {})),
            batch_size=config_dict.get('batch_size', 10),
            output_base_dir=config_dict.get('output_base_dir', 'output')
        )