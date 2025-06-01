"""Configuration modules for dependency injection."""

import os
from injector import Module, provider, singleton

from apiana.configuration import (
    ChatGPTExportProcessorConfiguration,
    PromptConfig,
    LLMProviderConfig,
    EmbedderConfig,
    Neo4jConfig
)


class BaseConfigModule(Module):
    """Base configuration module with default providers.
    
    This module provides the foundation for all configuration and can be
    extended by environment-specific modules that override specific providers.
    """
    
    @singleton
    @provider
    def provide_prompt_config(self) -> PromptConfig:
        """Provide default prompt configuration."""
        return PromptConfig(
            name="default",
            system_prompt_file="prompts/self-reflective-system-message.txt",
            user_prompt_template_file="prompts/self-reflective-prompt-template.txt"
        )
    
    @singleton
    @provider
    def provide_llm_config(self) -> LLMProviderConfig:
        """Provide default LLM configuration."""
        return LLMProviderConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=4096
        )
    
    @singleton
    @provider
    def provide_embedder_config(self) -> EmbedderConfig:
        """Provide default embedder configuration."""
        return EmbedderConfig(
            type="local",
            model="nomic-ai/nomic-embed-text-v1",
            dimension=768
        )
    
    @singleton
    @provider
    def provide_neo4j_config(self) -> Neo4jConfig:
        """Provide default Neo4j configuration."""
        return Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )
    
    @singleton
    @provider
    def provide_processor_config(
        self,
        prompt: PromptConfig,
        llm_provider: LLMProviderConfig,
        embedder: EmbedderConfig,
        neo4j: Neo4jConfig
    ) -> ChatGPTExportProcessorConfiguration:
        """Provide the main processor configuration."""
        return ChatGPTExportProcessorConfiguration(
            environment=self._get_environment(),
            batch_size=self._get_batch_size(),
            output_base_dir=self._get_output_dir(),
            prompt=prompt,
            llm_provider=llm_provider,
            embedder=embedder,
            neo4j=neo4j
        )
    
    def _get_environment(self) -> str:
        """Get environment name. Override in subclasses."""
        return "base"
    
    def _get_batch_size(self) -> int:
        """Get batch size. Override in subclasses."""
        return 10
    
    def _get_output_dir(self) -> str:
        """Get output directory. Override in subclasses."""
        return "output"


class LocalConfigModule(BaseConfigModule):
    """Configuration for local development with Ollama."""
    
    def _get_environment(self) -> str:
        return "local"
    
    @singleton
    @provider
    def provide_llm_config(self) -> LLMProviderConfig:
        """Local Ollama configuration."""
        return LLMProviderConfig(
            model="llama3.2",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.7,
            max_tokens=4096
        )


class DevConfigModule(BaseConfigModule):
    """Configuration for development environment."""
    
    def _get_environment(self) -> str:
        return "dev"
    
    @singleton
    @provider
    def provide_llm_config(self) -> LLMProviderConfig:
        """Remote Ollama configuration for dev."""
        return LLMProviderConfig(
            model=os.getenv("APIANA_LLM_MODEL", "llama3.2"),
            base_url=os.getenv("APIANA_LLM_BASE_URL", "http://dev-ollama:11434/v1"),
            api_key=os.getenv("APIANA_LLM_API_KEY", "ollama"),
            temperature=float(os.getenv("APIANA_LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("APIANA_LLM_MAX_TOKENS", "4096"))
        )
    
    @singleton
    @provider
    def provide_neo4j_config(self) -> Neo4jConfig:
        """Dev Neo4j configuration."""
        return Neo4jConfig(
            uri=os.getenv("APIANA_NEO4J_URI", "bolt://dev-neo4j:7687"),
            username=os.getenv("APIANA_NEO4J_USERNAME", "neo4j"),
            password=os.getenv("APIANA_NEO4J_PASSWORD", "dev-password"),
            database=os.getenv("APIANA_NEO4J_DATABASE")
        )


class ProductionConfigModule(BaseConfigModule):
    """Configuration for production environment with OpenAI."""
    
    def _get_environment(self) -> str:
        return "production"
    
    def _get_batch_size(self) -> int:
        return int(os.getenv("APIANA_BATCH_SIZE", "20"))
    
    @singleton
    @provider
    def provide_llm_config(self) -> LLMProviderConfig:
        """OpenAI configuration for production."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for production")
        
        return LLMProviderConfig(
            model=os.getenv("APIANA_LLM_MODEL", "gpt-4-turbo-preview"),
            api_key=api_key,
            temperature=float(os.getenv("APIANA_LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("APIANA_LLM_MAX_TOKENS", "4096"))
        )
    
    @singleton
    @provider
    def provide_embedder_config(self) -> EmbedderConfig:
        """OpenAI embeddings for production."""
        return EmbedderConfig(
            type="openai",
            model="text-embedding-3-small",
            dimension=1536
        )
    
    @singleton
    @provider
    def provide_neo4j_config(self) -> Neo4jConfig:
        """fcdvxsza"""
        uri = os.getenv("APIANA_NEO4J_URI")
        username = os.getenv("APIANA_NEO4J_USERNAME")
        password = os.getenv("APIANA_NEO4J_PASSWORD")
        
        if not all([uri, username, password]):
            raise ValueError("Neo4j configuration required for production")
        
        return Neo4jConfig(
            uri=uri,
            username=username,
            password=password,
            database=os.getenv("APIANA_NEO4J_DATABASE")
        )