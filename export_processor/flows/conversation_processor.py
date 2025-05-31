"""
Prefect workflow for processing ChatGPT conversations into experiential summaries.
Simplified version focusing on summary generation and storage.
"""

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from export_processor.chatgpt.schema import OpenAIConversation
from export_processor.llm.ollama_client import OllamaClient
from export_processor.storage.neo4j_store import Neo4jMemoryStore


# ============================================================================
# Summary Generation Tasks
# ============================================================================

@task(name="generate_experiential_summary", retries=3, retry_delay_seconds=10)
def generate_experiential_summary(
    conversation: OpenAIConversation,
    prompt_template: str,
    ollama_client: OllamaClient
) -> Dict[str, Any]:
    """
    Generate first-person experiential summary of conversation.
    
    Args:
        conversation: The conversation to summarize
        prompt_template: Template for generating summaries (should encourage first-person)
        ollama_client: Ollama client for LLM calls
        
    Returns:
        Dictionary containing:
        - conversation_id: Original conversation ID
        - title: Conversation title
        - content: First-person narrative summary
        - message_count: Number of messages
        - created_at: Original conversation timestamp
        - summary_timestamp: When summary was generated
    """
    # TODO: Implement Ollama call with proper prompt engineering
    # TODO: Format messages for context window
    # TODO: Extract emotional tone if needed
    pass


@task(name="extract_contextual_tags", retries=2)
def extract_contextual_tags(
    summary_content: str,
    conversation_title: str,
    ollama_client: OllamaClient
) -> Dict[str, Any]:
    """
    Extract contextual tags from the summary using a separate agent.
    
    This task analyzes the experiential summary to extract:
    - Emotional context (user's emotional state)
    - Environmental context (location, time, weather)
    - Activity context (what the user was doing)
    - Social context (type of interaction)
    
    Args:
        summary_content: The experiential summary text
        conversation_title: Title for additional context
        ollama_client: Ollama client for analysis
        
    Returns:
        Dictionary with contextual tags:
        - emotional_context: Primary and secondary emotions with intensity
        - environmental_context: Location, time of day, weather (inferred)
        - activity_context: Activities, tools, domains
        - social_context: Interaction type, formality, audience
    """
    # TODO: Implement contextual extraction with specialized prompt
    # TODO: Return structured context data
    # Placeholder return
    return {
        'emotional_context': None,
        'environmental_context': None,
        'activity_context': None,
        'social_context': None
    }


@task(name="generate_embedding")
def generate_embedding(
    text: str,
    embedding_client: Any  # TODO: Define embedding client type
) -> List[float]:
    """
    Generate embedding vector for text.
    
    Args:
        text: Text to embed (usually the summary)
        embedding_client: Client for embedding model
        
    Returns:
        Embedding vector as list of floats
    """
    # TODO: Implement embedding generation
    # TODO: Handle different embedding models (Ollama vs dedicated)
    pass


@task(name="store_summary_in_neo4j", retries=2)
def store_summary_in_neo4j(
    summary: Dict[str, Any],
    embedding: List[float],
    neo4j_store: Neo4jMemoryStore
) -> str:
    """
    Store summary and embedding in Neo4j with contextual tags.
    
    Creates:
    - ExperientialSummary node with content, embedding, and context
    
    Args:
        summary: Dictionary containing summary and contextual data
        embedding: Vector embedding
        neo4j_store: Neo4j store instance
    
    Returns:
        memory_id: The ID of the created memory node
    """
    # TODO: Implement actual storage with contextual data
    # Will call neo4j_store.create_experiential_summary()
    # Need to handle the contextual tags separately
    pass


@task(name="write_summary_to_disk")
def write_summary_to_disk(
    conversation_id: str,
    title: str,
    summary: str,
    output_dir: Path
) -> Path:
    """
    Write summary to disk for archival/backup.
    
    Args:
        conversation_id: ID of original conversation
        title: Conversation title
        summary: Generated summary text
        output_dir: Directory to write summaries
        
    Returns:
        Path to written file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from conversation ID and title (sanitized)
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))[:50]
    filename = f"{conversation_id}_{safe_title}.md"
    output_path = output_dir / filename
    
    # Write markdown file with metadata
    content = f"""---
conversation_id: {conversation_id}
title: {title}
generated_at: {datetime.now().isoformat()}
---

# {title}

{summary}
"""
    
    output_path.write_text(content, encoding='utf-8')
    return output_path


# ============================================================================
# Single Conversation Flow
# ============================================================================

@flow(name="process_single_conversation")
def process_single_conversation_flow(
    conversation: OpenAIConversation,
    prompt_template: str,
    ollama_client: OllamaClient,
    embedding_client: Any,
    neo4j_store: Neo4jMemoryStore,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Process a single conversation through summary generation pipeline.
    
    Returns:
        Dictionary with processing results
    """
    try:
        # Generate summary
        summary = generate_experiential_summary(
            conversation, prompt_template, ollama_client
        )
        
        # Extract contextual tags (can be done in parallel with disk write)
        contextual_tags = extract_contextual_tags(
            summary['content'],
            conversation.title,
            ollama_client
        )
        
        # Write to disk first (before Neo4j)
        file_path = write_summary_to_disk(
            conversation.id,
            conversation.title,
            summary['content'],
            output_dir
        )
        
        # Generate embedding
        embedding = generate_embedding(
            summary['content'], embedding_client
        )
        
        # Merge contextual tags into summary data
        summary.update(contextual_tags)
        
        # Store in Neo4j with all context
        memory_id = store_summary_in_neo4j(
            summary, embedding, neo4j_store
        )
        
        return {
            'status': 'success',
            'conversation_id': conversation.id,
            'memory_id': memory_id,
            'file_path': str(file_path),
            'summary_length': len(summary['content'])
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'conversation_id': conversation.id,
            'error': str(e)
        }


# ============================================================================
# Main Batch Processing Flow
# ============================================================================

@flow(
    name="batch_process_conversations",
    task_runner=ConcurrentTaskRunner(max_workers=5)
)
def batch_process_conversations_flow(
    conversations: List[OpenAIConversation],
    prompt_template: str,
    output_dir: Path,
    ollama_model: str = "llama3.2",
    embedding_model: str = "nomic-embed-text",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_auth: tuple = ("neo4j", "password"),
    batch_size: int = 10,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Main flow for batch processing conversations.
    
    Args:
        conversations: List of conversations from TUI (already loaded)
        prompt_template: Template for summary generation
        output_dir: Directory for output files
        ollama_model: Model to use for summaries
        embedding_model: Model to use for embeddings
        neo4j_uri: Neo4j connection URI
        neo4j_auth: Neo4j authentication tuple
        batch_size: Number of conversations per batch
        progress_callback: Optional callback for progress updates
        
    Returns:
        Processing statistics
    """
    # Initialize clients
    ollama_client = OllamaClient(model=ollama_model)
    
    # TODO: Initialize embedding client based on model choice
    embedding_client = None  # Placeholder
    
    # Initialize Neo4j store
    neo4j_store = Neo4jMemoryStore(
        uri=neo4j_uri,
        auth=neo4j_auth,
        embedding_model=embedding_model
    )
    
    # Setup Neo4j schema if needed
    neo4j_store.setup_schema()
    
    total = len(conversations)
    results = []
    
    # Process in batches
    for i in range(0, total, batch_size):
        batch = conversations[i:i + batch_size]
        batch_futures = []
        
        # Submit batch for parallel processing
        for conv in batch:
            future = process_single_conversation_flow.submit(
                conv,
                prompt_template,
                ollama_client,
                embedding_client,
                neo4j_store,
                output_dir
            )
            batch_futures.append(future)
        
        # Collect results and update progress
        for j, future in enumerate(batch_futures):
            result = future.result()
            results.append(result)
            
            # Update progress if callback provided
            if progress_callback:
                completed = i + j + 1
                progress_callback(
                    completed=completed,
                    total=total,
                    current_item=batch[j].title,
                    status=result['status']
                )
    
    # Calculate statistics
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    return {
        'total_processed': len(results),
        'successful': successful,
        'failed': failed,
        'results': results,
        'output_directory': str(output_dir),
        'timestamp': datetime.now().isoformat()
    }


# ============================================================================
# Progress Tracking Bridge for TUI
# ============================================================================

class PrefectProgressBridge:
    """Bridge between Prefect flow and TUI progress updates."""
    
    def __init__(self, tui_app):
        self.tui_app = tui_app
        
    def update_progress(self, completed: int, total: int, current_item: str, status: str):
        """Send progress update to TUI."""
        # TODO: Implement message passing to TUI
        # This will depend on TUI's message system
        pass