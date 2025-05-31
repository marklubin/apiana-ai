"""
Prefect workflow for processing ChatGPT conversations into experiential summaries.
Simplified version focusing on summary generation and storage.
"""

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from apiana.applications.chatgpt_export_tui.schema import OpenAIConversation
from apiana.llm.openai_agent import LLMAgent
from apiana.storage.neo4j_store import Neo4jMemoryStore, ProcessorRun
from apiana.embeddings.local_embedder import LocalEmbedder
from apiana.configuration import ProcessorConfig


# ============================================================================
# File Loading Tasks
# ============================================================================

@task(name="load_prompt_template")
def load_prompt_template(template_path: Path) -> str:
    """
    Load prompt template from file.
    
    Args:
        template_path: Path to the template file
        
    Returns:
        Template string content
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    
    return template_path.read_text(encoding='utf-8')


# ============================================================================
# Run Management Tasks
# ============================================================================

@task(name="create_processor_run")
def create_processor_run(
    config: ProcessorConfig,
    total_conversations: int,
    neo4j_store: Neo4jMemoryStore
) -> ProcessorRun:
    """
    Create a ProcessorRun node in Neo4j to track this batch job.
    
    Args:
        config: Processor configuration
        total_conversations: Total number of conversations to process
        neo4j_store: Neo4j store instance
        
    Returns:
        ProcessorRun node instance
    """
    run = ProcessorRun(
        run_id=config.run_id,
        prompt_name=config.prompt.name,
        prompt_files={
            'system_prompt_file': config.prompt.system_prompt_file,
            'user_prompt_template_file': config.prompt.user_prompt_template_file
        },
        embedder_config=config.embedder.to_dict(),
        llm_provider_config=config.llm_provider.to_dict(),
        total_conversations=total_conversations,
        output_directory=str(config.get_output_dir())
    )
    run.save()
    return run


@task(name="update_processor_run")
def update_processor_run(
    run: ProcessorRun,
    successful: int,
    failed: int,
    duration_seconds: float,
    status: str = "completed"
) -> None:
    """
    Update ProcessorRun with final statistics.
    
    Args:
        run: ProcessorRun instance
        successful: Number of successful summaries
        failed: Number of failed summaries
        duration_seconds: Total run duration in seconds
        status: Final status
    """
    run.successful_summaries = successful
    run.failed_summaries = failed
    run.duration_seconds = duration_seconds
    run.status = status
    run.completed_at = datetime.now()
    run.save()


# ============================================================================
# Summary Generation Tasks
# ============================================================================

@task(name="generate_experiential_summary", retries=3, retry_delay_seconds=10)
def generate_experiential_summary(
    conversation: OpenAIConversation,
    user_prompt: str,
    llm_agent: LLMAgent
) -> Dict[str, Any]:
    """
    Generate first-person experiential summary of conversation.
    
    Args:
        conversation: The conversation to summarize
        user_prompt: The prompt to send to the LLM for summary generation
        llm_agent: LLM client for generating summaries
        
    Returns:
        Dictionary containing:
        - conversation_id: Original conversation ID
        - title: Conversation title
        - content: First-person narrative summary
        - message_count: Number of messages
        - created_at: Original conversation timestamp
        - summary_timestamp: When summary was generated
    """
    # Format conversation messages
    formatted_messages = []
    for msg in conversation.messages:
        if msg.role == "user":
            formatted_messages.append(f"Mark: {msg.content}")
        else:
            formatted_messages.append(f"You (as AI Assistant): {msg.content}")
    
    # Format the prompt with conversation data
    formatted_prompt = user_prompt.format(
        conversation_date=conversation.created_at.strftime('%B %d, %Y') if conversation.created_at else "unknown date",
        formatted_messages="\n".join(formatted_messages)
    )
    
    # Generate summary using LLM
    summary_content = llm_agent.generate(formatted_prompt)
    
    return {
        'conversation_id': conversation.id,
        'title': conversation.title,
        'content': summary_content,
        'message_count': len(conversation.messages),
        'created_at': conversation.created_at,
        'summary_timestamp': datetime.now()
    }


@task(name="extract_contextual_tags", retries=2)
def extract_contextual_tags(
    summary_content: str,
    conversation_title: str,
    llm_agent: LLMAgent
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
    embedder: 'LocalEmbedder'
) -> List[float]:
    """
    Generate embedding vector for text.
    
    Args:
        text: Text to embed (usually the summary)
        embedder: LocalEmbedder instance
        
    Returns:
        Embedding vector as list of floats
    """
    return embedder.embed(text)


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
    conversation: OpenAIConversation,
    summary: str,
    output_dir: Path
) -> Path:
    """
    Write summary to disk with date-based filename.
    
    Args:
        conversation: Original conversation object
        summary: Generated summary text
        output_dir: Directory to write summaries
        
    Returns:
        Path to written file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Format: YYYY-MM-DD_conversation_title.txt
    created_date = conversation.created_at.strftime("%Y-%m-%d") if conversation.created_at else "unknown"
    safe_title = "".join(c for c in conversation.title if c.isalnum() or c in (' ', '-', '_'))[:50].strip()
    filename = f"{created_date}_{safe_title}.txt"
    output_path = output_dir / filename
    
    # Write plain text file
    output_path.write_text(summary, encoding='utf-8')
    return output_path


# ============================================================================
# Single Conversation Flow
# ============================================================================

@flow(name="process_single_conversation")
def process_single_conversation_flow(
    conversation: OpenAIConversation,
    prompt_template: str,
    llm_agent: LLMAgent,
    embedder: LocalEmbedder,
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
            conversation, prompt_template, llm_agent
        )
        
        # Extract contextual tags (can be done in parallel with disk write)
        contextual_tags = extract_contextual_tags(
            summary['content'],
            conversation.title,
            llm_agent
        )
        
        # Write to disk first (before Neo4j)
        file_path = write_summary_to_disk(
            conversation,
            summary['content'],
            output_dir
        )
        
        # Generate embedding
        embedding = generate_embedding(
            summary['content'], embedder
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
    config: ProcessorConfig,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Main flow for batch processing conversations with configuration.
    
    Args:
        conversations: List of conversations from TUI (already loaded)
        config: Processor configuration object
        progress_callback: Optional callback for progress updates
        
    Returns:
        Processing statistics
    """
    # Track start time
    start_time = datetime.now()
    
    # Create output directories
    output_dir = config.get_output_dir()
    summaries_dir = config.get_summaries_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration to run directory
    config.save_to_file(output_dir / "run_config.json")
    
    # Load prompt files
    prompts_dir = Path("prompts")
    system_prompt = load_prompt_template(prompts_dir / config.prompt.system_prompt_file)
    user_prompt_template = load_prompt_template(prompts_dir / config.prompt.user_prompt_template_file)
    
    # Initialize clients based on config
    llm_agent = LLMAgent(
        model=config.llm_provider.model,
        system_prompt=system_prompt,  # Use loaded system prompt
        base_url=config.llm_provider.base_url,
        api_key=config.llm_provider.api_key,
        temperature=config.llm_provider.temperature,
        max_tokens=config.llm_provider.max_tokens
    )
    
    # Initialize embedder based on config
    if config.embedder.type == "local":
        embedder = LocalEmbedder(model_name=config.embedder.model)
    else:
        # TODO: Add remote embedder support
        raise NotImplementedError("Remote embedders not yet supported")
    
    # Initialize Neo4j store
    neo4j_store = Neo4jMemoryStore(
        uri=config.neo4j.uri,
        auth=config.neo4j.auth,
        embedding_model=config.embedder.model,
        embedding_dimension=config.embedder.dimension
    )
    
    # Setup Neo4j schema if needed
    neo4j_store.setup_schema()
    
    # Create processor run in Neo4j
    processor_run = create_processor_run(
        config=config,
        total_conversations=len(conversations),
        neo4j_store=neo4j_store
    )
    
    total = len(conversations)
    results = []
    
    # Process in batches
    for i in range(0, total, config.batch_size):
        batch = conversations[i:i + config.batch_size]
        batch_futures = []
        
        # Submit batch for parallel processing
        for conv in batch:
            future = process_single_conversation_flow.submit(
                conv,
                user_prompt_template,  # Use loaded template
                llm_agent,
                embedder,
                neo4j_store,
                summaries_dir  # Use summaries subdirectory
            )
            batch_futures.append(future)
        
        # Collect results as they complete (non-blocking)
        from concurrent.futures import as_completed
        
        # Create mapping of future to conversation for progress tracking
        future_to_conv = {future: conv for future, conv in zip(batch_futures, batch)}
        
        # Process futures as they complete
        for future in as_completed(batch_futures):
            try:
                result = future.result()
                results.append(result)
                
                # Update progress if callback provided
                if progress_callback:
                    completed = len(results)
                    conv = future_to_conv[future]
                    progress_callback(
                        completed=completed,
                        total=total,
                        current_item=conv.title,
                        status=result['status']
                    )
            except Exception as e:
                # Handle failed future
                conv = future_to_conv[future]
                results.append({
                    'status': 'failed',
                    'conversation_id': conv.id,
                    'error': str(e)
                })
                
                if progress_callback:
                    completed = len(results)
                    progress_callback(
                        completed=completed,
                        total=total,
                        current_item=conv.title,
                        status='failed'
                    )
    
    # Calculate statistics
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    # Calculate duration
    end_time = datetime.now()
    duration_seconds = (end_time - start_time).total_seconds()
    
    # Update processor run with final stats
    update_processor_run(
        run=processor_run,
        successful=successful,
        failed=failed,
        duration_seconds=duration_seconds,
        status="completed" if failed == 0 else "completed_with_errors"
    )
    
    return {
        'run_id': config.run_id,
        'total_processed': len(results),
        'successful': successful,
        'failed': failed,
        'duration_seconds': duration_seconds,
        'results': results,
        'output_directory': str(output_dir),
        'summaries_directory': str(summaries_dir),
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