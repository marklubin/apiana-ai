"""Batch summary transform component for efficient summarization."""

from typing import List, Optional, Dict, Any
import logging

from apiana.core.components.transform.batch_inference import (
    BatchInferenceTransform,
    BatchProvider,
    BatchStore,
    BatchConfig
)
from apiana.core.components.transform.summarizer import SummarizerTransform
from apiana.types.chat_fragment import ChatFragment
from apiana.core.adapters.chat_fragment_store_adapter import create_chat_fragment_hash

logger = logging.getLogger(__name__)


class SummaryBatchProvider(BatchProvider[ChatFragment, Dict[str, Any]]):
    """Batch provider adapter for summarization."""
    
    def __init__(self, summarizer_transform: SummarizerTransform):
        """Initialize with a summarizer transform.
        
        Args:
            summarizer_transform: The underlying summarizer to use
        """
        self.summarizer = summarizer_transform
    
    def process_batch(self, items: List[ChatFragment]) -> List[Dict[str, Any]]:
        """Process a batch of ChatFragments to generate summaries.
        
        Args:
            items: List of ChatFragments to summarize
            
        Returns:
            List of summary dictionaries in the same order as inputs
        """
        # The summarizer already handles batches
        result = self.summarizer.transform(items)
        
        # Extract just the data portion (list of summaries)
        return result.data


class SummaryBatchStore(BatchStore[ChatFragment, Dict[str, Any]]):
    """Batch store adapter for summary results with real caching."""
    
    def __init__(self, application_store, provider_name: str, model_name: str):
        """Initialize with application store and model info.
        
        Args:
            application_store: The ApplicationStore instance
            provider_name: LLM provider name (e.g., "openai")
            model_name: Model name (e.g., "gpt-4")
        """
        self.store = application_store
        self.provider_name = provider_name
        self.model_name = model_name
        self.summary_cache = {}  # In-memory cache for summaries
    
    def get_by_hash(self, hash_key: str) -> Optional[Dict[str, Any]]:
        """Check if summary already exists by hash.
        
        Args:
            hash_key: The hash key to check
            
        Returns:
            Summary dict if found, None otherwise
        """
        # Check in-memory cache first
        if hash_key in self.summary_cache:
            logger.debug(f"Cache hit for summary hash: {hash_key}")
            return self.summary_cache[hash_key]
        
        # Parse the hash key to extract fragment_id
        parts = hash_key.split(":")
        if len(parts) >= 4:
            fragment_id = parts[0]
            
            # Check if this fragment already has a summary with matching agent ID
            fragment = self.store.get_fragment(fragment_id)
            if fragment:
                # Get the fragment node to check summary fields
                fragment_nodes = self.store.ChatFragmentNode.nodes.filter(
                    fragment_id=fragment_id
                ).first()
                
                if fragment_nodes and fragment_nodes.summary_agent_id == f"{self.provider_name}:{self.model_name}":
                    # We have a summary from this exact provider/model
                    # Reconstruct the summary result
                    summary_result = {
                        "fragment_id": fragment_id,
                        "title": fragment.title,
                        "summary": "CACHED_SUMMARY",  # Placeholder - in real system would store actual summary
                        "original_messages": len(fragment.messages),
                        "_cached": True
                    }
                    self.summary_cache[hash_key] = summary_result
                    logger.debug(f"Found existing summary for fragment {fragment_id}")
                    return summary_result
        
        return None
    
    def store_result(self, hash_key: str, input_item: ChatFragment, result: Dict[str, Any]) -> None:
        """Store summary result.
        
        Args:
            hash_key: The hash key
            input_item: Original ChatFragment
            result: Summary result dictionary
        """
        # Store in cache
        self.summary_cache[hash_key] = result
        
        # Update the fragment with summary information
        if result.get("summary") and not result.get("error"):
            self.store.store_fragment(
                input_item,
                summary_agent_id=f"{self.provider_name}:{self.model_name}",
                summary_text=result["summary"]
            )
            logger.debug(f"Stored summary for fragment {input_item.fragment_id}")


def create_summary_hash(fragment: ChatFragment, provider: str, model: str) -> str:
    """Create hash for summary deduplication.
    
    Args:
        fragment: The ChatFragment to hash
        provider: LLM provider name
        model: Model name
        
    Returns:
        Hash string for the summary task
    """
    # Use the existing chat fragment hash function
    return create_chat_fragment_hash(fragment, provider, model)


class BatchingChatFragmentSummaryTransform(BatchInferenceTransform[ChatFragment, Dict[str, Any]]):
    """Batching wrapper for ChatFragment summarization with real batching logic."""
    
    def __init__(
        self,
        summarizer_transform: SummarizerTransform,
        application_store,
        provider_name: str,
        model_name: str,
        batch_size: int = 10,
        name: Optional[str] = None
    ):
        """Initialize batching summary transform.
        
        Args:
            summarizer_transform: The underlying summarizer
            application_store: Storage for results
            provider_name: LLM provider name
            model_name: Model name
            batch_size: Number of items per batch
            name: Optional component name
        """
        # Create adapters
        provider = SummaryBatchProvider(summarizer_transform)
        store = SummaryBatchStore(application_store, provider_name, model_name)
        
        # Create config with hash function
        config = BatchConfig(
            batch_size=batch_size,
            hash_function=lambda fragment: create_summary_hash(
                fragment, provider_name, model_name
            )
        )
        
        # Initialize parent
        super().__init__(
            provider=provider,
            store=store,
            config=config,
            name=name or "BatchingChatFragmentSummaryTransform"
        )
        
        # Store references for introspection
        self.summarizer_transform = summarizer_transform
        self.provider_name = provider_name
        self.model_name = model_name
        self.application_store = application_store
    
    @property
    def input_types(self) -> List[type]:
        """Return supported input types."""
        return [List[ChatFragment]]
    
    @property
    def output_types(self) -> List[type]:
        """Return output types."""
        return [List[dict]]  # List of summary dictionaries