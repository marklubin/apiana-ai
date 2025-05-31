"""
Pipeline components for ChatGPT conversation processing.
Uses neo4j-graphrag pipeline framework.
"""

from neo4j_graphrag.experimental.pipeline import Component, DataModel
from neo4j_graphrag.experimental.pipeline.types.context import RunContext
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from apiana.applications.chatgpt_export_tui.schema import OpenAIConversation
from apiana.llm.openai_agent import LLMAgent
from apiana.embeddings.local_embedder import LocalEmbedder
from apiana.storage.neo4j_store import Neo4jMemoryStore, ProcessorRun
from apiana.configuration import ProcessorConfig


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ConversationBatch(DataModel):
    """Batch of conversations to process."""
    conversations: List[OpenAIConversation]
    total_count: int
    config: ProcessorConfig


@dataclass
class ProcessedSummary(DataModel):
    """Individual processed summary with all enrichments."""
    conversation_id: str
    title: str
    summary_content: str
    embedding: List[float] = field(default_factory=list)
    contextual_tags: Dict[str, Any] = field(default_factory=dict)
    message_count: int = 0
    created_at: Optional[datetime] = None
    summary_timestamp: datetime = field(default_factory=datetime.now)
    file_path: Optional[str] = None


@dataclass
class BatchProcessingResult(DataModel):
    """Result of processing a batch of conversations."""
    summaries: List[ProcessedSummary]
    failed_conversations: List[Dict[str, str]]  # id -> error
    successful_count: int
    failed_count: int
    processor_run_id: str


# ============================================================================
# Components
# ============================================================================

class LoadPromptComponent(Component):
    """Load prompt templates from files."""
    
    def __init__(self, prompts_dir: Path = Path("prompts")):
        self.prompts_dir = prompts_dir
    
    async def run(self, config: ProcessorConfig) -> Dict[str, str]:
        """Load system and user prompts based on config."""
        system_prompt_path = self.prompts_dir / config.prompt.system_prompt_file
        user_prompt_path = self.prompts_dir / config.prompt.user_prompt_template_file
        
        if not system_prompt_path.exists():
            raise FileNotFoundError(f"System prompt not found: {system_prompt_path}")
        if not user_prompt_path.exists():
            raise FileNotFoundError(f"User prompt not found: {user_prompt_path}")
        
        return {
            "system_prompt": system_prompt_path.read_text(encoding='utf-8'),
            "user_prompt_template": user_prompt_path.read_text(encoding='utf-8')
        }


class CreateProcessorRunComponent(Component):
    """Create a ProcessorRun node in Neo4j to track this batch job."""
    
    def __init__(self, neo4j_store: Neo4jMemoryStore):
        self.neo4j_store = neo4j_store
    
    async def run(self, batch: ConversationBatch) -> ProcessorRun:
        """Create processor run tracking node."""
        config = batch.config
        run = ProcessorRun(
            run_id=config.run_id,
            prompt_name=config.prompt.name,
            prompt_files={
                'system_prompt_file': config.prompt.system_prompt_file,
                'user_prompt_template_file': config.prompt.user_prompt_template_file
            },
            embedder_config=config.embedder.to_dict(),
            llm_provider_config=config.llm_provider.to_dict(),
            total_conversations=batch.total_count,
            output_directory=str(config.get_output_dir())
        )
        run.save()
        return run


class GenerateSummariesComponent(Component):
    """Generate experiential summaries for conversations."""
    
    def __init__(self, llm_agent: LLMAgent):
        self.llm_agent = llm_agent
    
    async def run_with_context(
        self, 
        context_: RunContext,
        batch: ConversationBatch,
        prompts: Dict[str, str],
        processor_run: ProcessorRun
    ) -> BatchProcessingResult:
        """Process all conversations in batch."""
        summaries = []
        failed_conversations = {}
        
        total = len(batch.conversations)
        user_prompt_template = prompts["user_prompt_template"]
        
        for i, conversation in enumerate(batch.conversations):
            # Send progress notification
            await context_.notify(
                message=f"Processing conversation {i+1}/{total}: {conversation.title}",
                data={"current": i+1, "total": total, "conversation_id": conversation.id}
            )
            
            try:
                # Format conversation messages
                formatted_messages = []
                for msg in conversation.messages:
                    if msg.role == "user":
                        formatted_messages.append(f"Mark: {msg.content}")
                    else:
                        formatted_messages.append(f"You (as AI Assistant): {msg.content}")
                
                # Format the prompt with conversation data
                formatted_prompt = user_prompt_template.format(
                    conversation_date=conversation.created_at.strftime('%B %d, %Y') if conversation.created_at else "unknown date",
                    formatted_messages="\n".join(formatted_messages)
                )
                
                # Generate summary using LLM
                summary_content = self.llm_agent.generate(formatted_prompt)
                
                summary = ProcessedSummary(
                    conversation_id=conversation.id,
                    title=conversation.title,
                    summary_content=summary_content,
                    message_count=len(conversation.messages),
                    created_at=conversation.created_at,
                    summary_timestamp=datetime.now()
                )
                summaries.append(summary)
                
            except Exception as e:
                failed_conversations[conversation.id] = str(e)
                await context_.notify(
                    message=f"Failed to process conversation {conversation.id}: {e}",
                    data={"error": str(e), "conversation_id": conversation.id}
                )
        
        return BatchProcessingResult(
            summaries=summaries,
            failed_conversations=failed_conversations,
            successful_count=len(summaries),
            failed_count=len(failed_conversations),
            processor_run_id=processor_run.run_id
        )


class ExtractContextualTagsComponent(Component):
    """Extract contextual tags from summaries."""
    
    def __init__(self, llm_agent: LLMAgent):
        self.llm_agent = llm_agent
    
    async def run(self, batch_result: BatchProcessingResult) -> BatchProcessingResult:
        """Extract contextual tags for each summary."""
        # TODO: Implement contextual extraction with specialized prompt
        # For now, just pass through
        for summary in batch_result.summaries:
            summary.contextual_tags = {
                'emotional_context': None,
                'environmental_context': None,
                'activity_context': None,
                'social_context': None
            }
        return batch_result


class GenerateEmbeddingsComponent(Component):
    """Generate embeddings for summaries."""
    
    def __init__(self, embedder: LocalEmbedder):
        self.embedder = embedder
    
    async def run_with_context(
        self,
        context_: RunContext,
        batch_result: BatchProcessingResult
    ) -> BatchProcessingResult:
        """Generate embeddings for all summaries."""
        total = len(batch_result.summaries)
        
        for i, summary in enumerate(batch_result.summaries):
            await context_.notify(
                message=f"Generating embedding {i+1}/{total}",
                data={"current": i+1, "total": total}
            )
            
            try:
                summary.embedding = self.embedder.embed(summary.summary_content)
            except Exception as e:
                # Log but don't fail the whole batch
                await context_.notify(
                    message=f"Failed to generate embedding for {summary.conversation_id}: {e}",
                    data={"error": str(e)}
                )
        
        return batch_result


class WriteSummariesToDiskComponent(Component):
    """Write summaries to disk."""
    
    async def run_with_context(
        self,
        context_: RunContext,
        batch_result: BatchProcessingResult,
        config: ProcessorConfig
    ) -> BatchProcessingResult:
        """Write all summaries to disk."""
        output_dir = config.get_summaries_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for summary in batch_result.summaries:
            try:
                # Format: YYYY-MM-DD_conversation_title.txt
                created_date = summary.created_at.strftime("%Y-%m-%d") if summary.created_at else "unknown"
                safe_title = "".join(c for c in summary.title if c.isalnum() or c in (' ', '-', '_'))[:50].strip()
                filename = f"{created_date}_{safe_title}.txt"
                output_path = output_dir / filename
                
                # Write plain text file
                output_path.write_text(summary.summary_content, encoding='utf-8')
                summary.file_path = str(output_path)
                
                await context_.notify(
                    message=f"Wrote summary to {filename}",
                    data={"file_path": str(output_path)}
                )
                
            except Exception as e:
                await context_.notify(
                    message=f"Failed to write summary for {summary.conversation_id}: {e}",
                    data={"error": str(e)}
                )
        
        return batch_result


class StoreInNeo4jComponent(Component):
    """Store summaries in Neo4j."""
    
    def __init__(self, neo4j_store: Neo4jMemoryStore):
        self.neo4j_store = neo4j_store
    
    async def run_with_context(
        self,
        context_: RunContext,
        batch_result: BatchProcessingResult
    ) -> BatchProcessingResult:
        """Store all summaries in Neo4j."""
        # TODO: Implement actual storage with contextual data
        # For now, this is a placeholder
        for summary in batch_result.summaries:
            await context_.notify(
                message=f"Storing summary for {summary.conversation_id} in Neo4j",
                data={"conversation_id": summary.conversation_id}
            )
            # neo4j_store.create_experiential_summary(...)
        
        return batch_result


class UpdateProcessorRunComponent(Component):
    """Update ProcessorRun with final statistics."""
    
    def __init__(self, neo4j_store: Neo4jMemoryStore):
        self.neo4j_store = neo4j_store
    
    async def run(
        self,
        processor_run: ProcessorRun,
        batch_result: BatchProcessingResult,
        start_time: datetime
    ) -> BatchProcessingResult:
        """Update processor run with final stats."""
        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()
        
        processor_run.successful_summaries = batch_result.successful_count
        processor_run.failed_summaries = batch_result.failed_count
        processor_run.duration_seconds = duration_seconds
        processor_run.status = "completed" if batch_result.failed_count == 0 else "completed_with_errors"
        processor_run.completed_at = end_time
        processor_run.save()
        
        return batch_result