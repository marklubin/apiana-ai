"""
Pipeline Factory Definitions

This module defines all available pipeline factory functions that can be used
throughout the application. Each factory function takes specific inputs and
returns a configured Pipeline instance ready for execution.

Pipeline factories are automatically discovered by the Gradio application
and other interfaces to provide dynamic UI generation.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
import inspect

from apiana.core.pipelines.base import PipelineBuilder, Pipeline
from apiana.core.components import (
    ChatGPTExportReader,
    ValidationTransform,
    ConversationChunkerComponent,
    SummarizerTransform,
    EmbeddingTransform
)
from apiana.core.components.writers import (
    ChatFragmentWriter,
    MemoryBlockWriter,
)
from apiana.core.components.managers import PipelineRunManager
from apiana.core.providers.local import LocalTransformersLLM
from apiana.stores import ApplicationStore, AgentMemoryStore
from apiana.types.configuration import Neo4jConfig
from neo4j_graphrag.embeddings.sentence_transformers import SentenceTransformerEmbeddings


@dataclass
class PipelineMetadata:
    """Metadata about a pipeline factory function for UI generation."""
    name: str
    description: str
    category: str
    requires_files: bool = False
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    output_description: str = ""
    estimated_time: str = ""
    tags: List[str] = field(default_factory=list)


# Pipeline Factory Functions
# Each function should be decorated with metadata for UI discovery

def chatgpt_full_processing_pipeline(
    neo4j_config: Neo4jConfig,
    agent_id: str,
    input_file: Path,
    run_name: Optional[str] = None,
    llm_model: str = "microsoft/DialoGPT-small",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 5000,
    min_messages: int = 2,
    summary_max_length: int = 150
) -> Pipeline:
    """
    Complete ChatGPT export processing with full persistence and memory generation.
    
    This pipeline provides end-to-end processing:
    - Reads ChatGPT export JSON files
    - Stores ChatFragments in application database
    - Validates conversations (configurable minimum messages)
    - Chunks large conversations (configurable token limit)
    - Summarizes using local LLM
    - Generates embeddings using local models
    - Stores final memories in agent-specific database
    - Tracks complete execution metadata
    
    Args:
        neo4j_config: Neo4j database connection configuration
        agent_id: Unique identifier for the agent (creates dedicated database)
        input_file: Path to ChatGPT export JSON file
        run_name: Optional custom name for this pipeline run
        llm_model: HuggingFace model name for summarization
        embedding_model: Sentence transformers model for embeddings
        max_tokens: Maximum tokens per conversation chunk
        min_messages: Minimum messages required per conversation
        summary_max_length: Maximum length for generated summaries
        
    Returns:
        Configured Pipeline ready for execution
    """
    # Generate run name if not provided
    if not run_name:
        run_name = f"Full ChatGPT Processing: {input_file.stem}"
    
    # Create stores
    app_store = ApplicationStore(neo4j_config)
    agent_store = AgentMemoryStore(neo4j_config, agent_id)
    
    # Create providers
    local_llm = LocalTransformersLLM(
        model_name=llm_model,
        device="auto"
    )
    
    local_embedder = SentenceTransformerEmbeddings(
        embedding_model,
        trust_remote_code=True
    )
    
    # Create pipeline manager for execution tracking
    pipeline_manager = PipelineRunManager(
        store=app_store,
        run_name=run_name,
        config={
            "agent_id": agent_id, 
            "input_file": str(input_file),
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "max_tokens": max_tokens,
            "min_messages": min_messages,
        },
        auto_link_fragments=True,
        name="pipeline_tracker"
    )
    
    # Create processing components
    reader = ChatGPTExportReader("chatgpt_reader")
    
    fragment_writer = ChatFragmentWriter(
        store=app_store,
        tags=["chatgpt_export", "full_processing", agent_id],
        name="fragment_persister"
    )
    
    validator = ValidationTransform(
        name="validator",
        config={"min_messages": min_messages}
    )
    
    chunker = ConversationChunkerComponent(
        name="chunker",
        config={"max_tokens": max_tokens}
    )
    
    summarizer = SummarizerTransform(
        name="summarizer",
        llm_provider=local_llm,
        config={"max_length": summary_max_length}
    )
    
    embedder = EmbeddingTransform(
        name="embedder",
        embedding_provider=local_embedder
    )
    
    memory_writer = MemoryBlockWriter(
        store=agent_store,
        agent_id=agent_id,
        tags=["conversation", "chatgpt_export", "full_processing"],
        name="memory_persister"
    )
    
    # Build the pipeline
    pipeline = (
        PipelineBuilder("ChatGPT Full Processing")
        .add_component(pipeline_manager)
        .add_component(reader)
        .add_component(fragment_writer)
        .add_component(validator)
        .add_component(chunker)
        .add_component(summarizer)
        .add_component(embedder)
        .add_component(memory_writer)
        .build()
    )
    
    return pipeline


def chatgpt_fragment_only_pipeline(
    neo4j_config: Neo4jConfig,
    input_file: Path,
    run_name: Optional[str] = None,
    min_messages: int = 2,
    max_tokens: int = 5000,
    tags: Optional[List[str]] = None
) -> Pipeline:
    """
    Simple ChatGPT processing that only stores ChatFragments without memory generation.
    
    This pipeline is ideal for initial data ingestion:
    - Reads ChatGPT export JSON files
    - Validates conversations
    - Chunks large conversations
    - Stores ChatFragments in application database
    - Tracks execution metadata
    
    Args:
        neo4j_config: Neo4j database connection configuration
        input_file: Path to ChatGPT export JSON file
        run_name: Optional custom name for this pipeline run
        min_messages: Minimum messages required per conversation
        max_tokens: Maximum tokens per conversation chunk
        tags: Optional custom tags for stored fragments
        
    Returns:
        Configured Pipeline for fragment storage only
    """
    if not run_name:
        run_name = f"Fragment Storage: {input_file.stem}"
    
    # Create application store
    app_store = ApplicationStore(neo4j_config)
    
    # Create pipeline manager
    pipeline_manager = PipelineRunManager(
        store=app_store,
        run_name=run_name,
        config={
            "input_file": str(input_file),
            "min_messages": min_messages,
            "max_tokens": max_tokens,
        },
        auto_link_fragments=True,
        name="pipeline_tracker"
    )
    
    # Create components
    reader = ChatGPTExportReader("chatgpt_reader")
    
    fragment_writer = ChatFragmentWriter(
        store=app_store,
        tags=tags or ["chatgpt_export", "fragment_only"],
        name="fragment_storage"
    )
    
    validator = ValidationTransform(
        name="validator",
        config={"min_messages": min_messages}
    )
    
    chunker = ConversationChunkerComponent(
        name="chunker",
        config={"max_tokens": max_tokens}
    )
    
    # Build pipeline
    pipeline = (
        PipelineBuilder("ChatGPT Fragment Storage")
        .add_component(pipeline_manager)
        .add_component(reader)
        .add_component(fragment_writer)
        .add_component(validator)
        .add_component(chunker)
        .build()
    )
    
    return pipeline


def fragment_to_memory_pipeline(
    neo4j_config: Neo4jConfig,
    agent_id: str,
    run_name: Optional[str] = None,
    llm_model: str = "microsoft/DialoGPT-small",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    summary_max_length: int = 150,
    memory_tags: Optional[List[str]] = None
) -> Pipeline:
    """
    Process existing ChatFragments from the database into agent memories.
    
    This pipeline is useful for post-processing stored ChatFragments:
    - Retrieves ChatFragments from application database
    - Summarizes conversations using local LLM
    - Generates embeddings using local models
    - Stores as agent memories in dedicated database
    
    Args:
        neo4j_config: Neo4j database connection configuration
        agent_id: Unique identifier for the agent
        run_name: Optional custom name for this pipeline run
        llm_model: HuggingFace model name for summarization
        embedding_model: Sentence transformers model for embeddings
        summary_max_length: Maximum length for generated summaries
        memory_tags: Optional custom tags for stored memories
        
    Returns:
        Pipeline for converting ChatFragments to agent memories
    """
    if not run_name:
        run_name = f"Memory Generation for Agent: {agent_id}"
    
    # Create stores
    app_store = ApplicationStore(neo4j_config)
    agent_store = AgentMemoryStore(neo4j_config, agent_id)
    
    # Create providers
    local_llm = LocalTransformersLLM(
        model_name=llm_model,
        device="auto"
    )
    
    local_embedder = SentenceTransformerEmbeddings(
        embedding_model,
        trust_remote_code=True
    )
    
    # Create pipeline manager
    pipeline_manager = PipelineRunManager(
        store=app_store,
        run_name=run_name,
        config={
            "agent_id": agent_id,
            "mode": "memory_generation",
            "llm_model": llm_model,
            "embedding_model": embedding_model,
        },
        name="pipeline_tracker"
    )
    
    # Create processing components
    summarizer = SummarizerTransform(
        name="summarizer",
        llm_provider=local_llm,
        config={"max_length": summary_max_length}
    )
    
    embedder = EmbeddingTransform(
        name="embedder",
        embedding_provider=local_embedder
    )
    
    memory_writer = MemoryBlockWriter(
        store=agent_store,
        agent_id=agent_id,
        tags=memory_tags or ["conversation", "processed", "memory_generation"],
        name="memory_writer"
    )
    
    # Build pipeline
    pipeline = (
        PipelineBuilder("Fragment to Memory Processing")
        .add_component(pipeline_manager)
        .add_component(summarizer)
        .add_component(embedder)
        .add_component(memory_writer)
        .build()
    )
    
    return pipeline


def dummy_test_pipeline(
    message: str = "Hello from dummy pipeline!",
    iterations: int = 3,
    delay_seconds: float = 0.5,
    output_file: Optional[Path] = None
) -> Pipeline:
    """
    Dummy pipeline for UI automation testing.
    
    This pipeline performs safe, non-impactful operations like printing messages,
    doing simple calculations, and creating temporary output files. Perfect for
    testing the UI without touching real data or external services.
    
    Args:
        message: Message to process and display
        iterations: Number of processing iterations to simulate
        delay_seconds: Delay between iterations to simulate work
        output_file: Optional file to write results to
        
    Returns:
        Configured Pipeline for testing
    """
    import time
    import json
    import logging
    from datetime import datetime
    from apiana.core.components.common import ComponentResult
    from apiana.core.components.common.base import Component
    
    logger = logging.getLogger(__name__)
    
    class DummyProcessor(Component):
        """Dummy component that performs safe test operations."""
        
        def __init__(self, name: str = "dummy_processor", config: dict = None):
            super().__init__(name, config)
            self.processed_items = []
            self.logger = logger
        
        def process(self, input_data) -> ComponentResult:
            """Process dummy data with simulated work."""
            start_time = time.time()
            
            message = self.config.get("message", "Default test message")
            iterations = self.config.get("iterations", 3)
            delay = self.config.get("delay_seconds", 0.5)
            
            self.logger.info(f"🎭 Starting dummy processing: {message}")
            
            results = []
            for i in range(iterations):
                self.logger.info(f"  🔄 Processing iteration {i+1}/{iterations}")
                
                # Simulate some work
                time.sleep(delay)
                
                # Create dummy result
                result = {
                    "iteration": i + 1,
                    "timestamp": datetime.now().isoformat(),
                    "message": f"{message} - Iteration {i+1}",
                    "random_number": (i + 1) * 42,
                    "status": "success"
                }
                results.append(result)
                self.processed_items.append(result)
                
                self.logger.info(f"    ✅ Completed iteration {i+1}")
            
            # Save to output file if specified
            output_file = self.config.get("output_file")
            if output_file:
                try:
                    output_path = Path(output_file)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_path, 'w') as f:
                        json.dump({
                            "summary": {
                                "total_iterations": iterations,
                                "message": message,
                                "completed_at": datetime.now().isoformat()
                            },
                            "results": results
                        }, f, indent=2)
                    
                    self.logger.info(f"💾 Results saved to: {output_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not save to file: {e}")
            
            execution_time = (time.time() - start_time) * 1000
            
            self.logger.info(f"🎉 Dummy processing completed! Processed {len(results)} items")
            
            return ComponentResult(
                data=results,
                metadata={
                    "total_iterations": iterations,
                    "message": message,
                    "output_file": str(output_file) if output_file else None,
                    "items_processed": len(results)
                },
                execution_time_ms=execution_time
            )
    
    # Build the dummy pipeline
    pipeline = (
        PipelineBuilder("Dummy Test Pipeline")
        .add_component(DummyProcessor("dummy_processor", {
            "message": message,
            "iterations": iterations,
            "delay_seconds": delay_seconds,
            "output_file": output_file
        }))
        .build()
    )
    
    return pipeline


# Pipeline Metadata Registry
# This allows the Gradio app to automatically discover and display pipelines

PIPELINE_REGISTRY = {
    "chatgpt_full_processing_pipeline": PipelineMetadata(
        name="ChatGPT Full Processing",
        description="Complete end-to-end processing of ChatGPT exports with memory generation",
        category="ChatGPT Processing",
        requires_files=True,
        input_parameters={
            "neo4j_config": {"type": "neo4j_config", "required": True},
            "agent_id": {"type": "string", "required": True, "description": "Unique agent identifier"},
            "input_file": {"type": "file", "required": True, "accept": ".json"},
            "run_name": {"type": "string", "required": False, "description": "Custom run name"},
            "llm_model": {"type": "string", "default": "microsoft/DialoGPT-small", "description": "HuggingFace model for summarization"},
            "embedding_model": {"type": "string", "default": "sentence-transformers/all-MiniLM-L6-v2", "description": "Embedding model"},
            "max_tokens": {"type": "integer", "default": 5000, "description": "Maximum tokens per chunk"},
            "min_messages": {"type": "integer", "default": 2, "description": "Minimum messages per conversation"},
            "summary_max_length": {"type": "integer", "default": 150, "description": "Maximum summary length"}
        },
        output_description="ChatFragments stored in application DB, memories stored in agent-specific DB",
        estimated_time="2-10 minutes depending on file size",
        tags=["chatgpt", "full-processing", "memory-generation", "ai-agent"]
    ),
    
    "chatgpt_fragment_only_pipeline": PipelineMetadata(
        name="ChatGPT Fragment Storage",
        description="Simple ChatGPT export processing that only stores conversation fragments",
        category="ChatGPT Processing",
        requires_files=True,
        input_parameters={
            "neo4j_config": {"type": "neo4j_config", "required": True},
            "input_file": {"type": "file", "required": True, "accept": ".json"},
            "run_name": {"type": "string", "required": False, "description": "Custom run name"},
            "min_messages": {"type": "integer", "default": 2, "description": "Minimum messages per conversation"},
            "max_tokens": {"type": "integer", "default": 5000, "description": "Maximum tokens per chunk"},
            "tags": {"type": "list", "required": False, "description": "Custom tags for fragments"}
        },
        output_description="ChatFragments stored in application database",
        estimated_time="30 seconds - 2 minutes",
        tags=["chatgpt", "storage", "fragments"]
    ),
    
    "fragment_to_memory_pipeline": PipelineMetadata(
        name="Fragment to Memory Conversion",
        description="Convert existing ChatFragments into agent-specific memories",
        category="Memory Processing",
        requires_files=False,
        input_parameters={
            "neo4j_config": {"type": "neo4j_config", "required": True},
            "agent_id": {"type": "string", "required": True, "description": "Target agent identifier"},
            "run_name": {"type": "string", "required": False, "description": "Custom run name"},
            "llm_model": {"type": "string", "default": "microsoft/DialoGPT-small", "description": "HuggingFace model for summarization"},
            "embedding_model": {"type": "string", "default": "sentence-transformers/all-MiniLM-L6-v2", "description": "Embedding model"},
            "summary_max_length": {"type": "integer", "default": 150, "description": "Maximum summary length"},
            "memory_tags": {"type": "list", "required": False, "description": "Custom tags for memories"}
        },
        output_description="Agent memories stored in agent-specific database",
        estimated_time="1-5 minutes depending on fragment count",
        tags=["memory-generation", "ai-agent", "post-processing"]
    ),
    
    "dummy_test_pipeline": PipelineMetadata(
        name="Dummy Test Pipeline",
        description="Safe testing pipeline that performs non-impactful operations like logging and simple calculations",
        category="Testing",
        requires_files=False,
        input_parameters={
            "message": {"type": "string", "default": "Hello from dummy pipeline!", "description": "Message to process"},
            "iterations": {"type": "integer", "default": 3, "description": "Number of processing iterations"},
            "delay_seconds": {"type": "number", "default": 0.5, "description": "Delay between iterations (seconds)"},
            "output_file": {"type": "file", "required": False, "description": "Optional output file path"}
        },
        output_description="JSON results with processing logs and dummy data",
        estimated_time="5-15 seconds",
        tags=["testing", "dummy", "safe", "ui-automation"]
    )
}


def get_available_pipelines() -> Dict[str, PipelineMetadata]:
    """
    Get all available pipeline factory functions with their metadata.
    
    Returns:
        Dictionary mapping function names to their metadata
    """
    return PIPELINE_REGISTRY.copy()


def get_pipeline_factory(pipeline_name: str):
    """
    Get a pipeline factory function by name.
    
    Args:
        pipeline_name: Name of the pipeline factory function
        
    Returns:
        Pipeline factory function
        
    Raises:
        ValueError: If pipeline name is not found
    """
    # Get the function from global namespace
    if pipeline_name in globals():
        return globals()[pipeline_name]
    else:
        raise ValueError(f"Pipeline '{pipeline_name}' not found")


def get_pipeline_signature(pipeline_name: str) -> inspect.Signature:
    """
    Get the function signature for a pipeline factory.
    
    Args:
        pipeline_name: Name of the pipeline factory function
        
    Returns:
        Function signature for parameter introspection
    """
    factory_func = get_pipeline_factory(pipeline_name)
    return inspect.signature(factory_func)