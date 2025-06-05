"""Batch embedding transform component for efficient embedding generation."""

from typing import List, Optional, Dict, Any
import logging

from apiana.core.components.transform.batch_inference import (
    BatchInferenceTransform,
    BatchProvider,
    BatchStore,
    BatchConfig
)
from apiana.core.components.transform.embedder import EmbeddingTransform
from apiana.stores.neo4j.application_store import create_embedding_hash

logger = logging.getLogger(__name__)


class EmbeddingBatchProvider(BatchProvider[Dict[str, Any], Dict[str, Any]]):
    """Batch provider adapter for embedding generation."""
    
    def __init__(self, embedding_transform: EmbeddingTransform):
        """Initialize with an embedding transform.
        
        Args:
            embedding_transform: The underlying embedder to use
        """
        self.embedder = embedding_transform
    
    def process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of summary dicts to generate embeddings.
        
        Args:
            items: List of summary dictionaries
            
        Returns:
            List of dictionaries with embeddings in the same order as inputs
        """
        # The embedder already handles batches
        result = self.embedder.transform(items)
        
        # Extract just the data portion (list of embedding results)
        return result.data


class EmbeddingBatchStore(BatchStore[Dict[str, Any], Dict[str, Any]]):
    """Batch store adapter for embedding results with real caching."""
    
    def __init__(self, application_store, provider_name: str, model_name: str):
        """Initialize with application store and model info.
        
        Args:
            application_store: The ApplicationStore instance
            provider_name: Embedding provider name (e.g., "openai")
            model_name: Model name (e.g., "text-embedding-ada-002")
        """
        self.store = application_store
        self.provider_name = provider_name
        self.model_name = model_name
        self.embedding_cache = {}  # In-memory cache
    
    def get_by_hash(self, hash_key: str) -> Optional[Dict[str, Any]]:
        """Check if embedding already exists by hash.
        
        Args:
            hash_key: The hash key to check
            
        Returns:
            Embedding dict if found, None otherwise
        """
        # Check in-memory cache first
        if hash_key in self.embedding_cache:
            logger.debug(f"Cache hit for embedding hash: {hash_key}")
            return self.embedding_cache[hash_key]
        
        # Parse the hash key to extract fragment_id
        parts = hash_key.split(":")
        if len(parts) >= 2:
            fragment_id = parts[0]
            
            # For summary items, we need to check with the source text
            # to verify the hash matches
            summary_item = getattr(self, '_current_item_map', {}).get(hash_key)
            source_text = summary_item.get("summary") if summary_item else None
            
            # Check if embedding exists
            embedding = self.store.get_embedding(
                source_id=fragment_id,
                model_provider=self.provider_name,
                model_name=self.model_name,
                source_text=source_text  # Pass source text for hash verification
            )
            
            if embedding:
                # Return in the expected format, including all original fields
                result = {
                    "fragment_id": fragment_id,
                    "embedding": embedding["vector"],
                    "embedding_dim": embedding["model_dimensions"],
                    "_cached": True
                }
                
                # Include other fields from the original item if available
                if summary_item:
                    result.update({
                        "title": summary_item.get("title"),
                        "summary": summary_item.get("summary"),
                        "original_messages": summary_item.get("original_messages")
                    })
                
                self.embedding_cache[hash_key] = result
                logger.debug(f"Found existing embedding for fragment {fragment_id}")
                return result
        
        return None
    
    def store_result(self, hash_key: str, input_item: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Store embedding result.
        
        Args:
            hash_key: The hash key
            input_item: Original summary dictionary
            result: Embedding result dictionary
        """
        # Store in cache
        self.embedding_cache[hash_key] = result
        
        # Store the embedding if it was generated successfully
        if result.get("embedding") and input_item.get("summary") and not result.get("error"):
            self.store.store_embedding(
                source_id=input_item["fragment_id"],
                source_type="summary",
                source_text=input_item["summary"],
                embedding_vector=result["embedding"],
                model_provider=self.provider_name,
                model_name=self.model_name,
                metadata={
                    "title": input_item.get("title"),
                    "original_messages": input_item.get("original_messages")
                }
            )
            logger.debug(f"Stored embedding for fragment {input_item['fragment_id']}")


def create_embedding_task_hash(summary_item: Dict[str, Any], provider: str, model: str) -> str:
    """Create hash for embedding task deduplication.
    
    Args:
        summary_item: Dictionary with summary data
        provider: Embedding provider name
        model: Model name
        
    Returns:
        Hash string for the embedding task
    """
    fragment_id = summary_item.get("fragment_id", "unknown")
    summary_text = summary_item.get("summary", "")
    
    # Create a composite hash
    text_hash = create_embedding_hash(summary_text, provider, model)
    return f"{fragment_id}:{text_hash}"


class BatchEmbeddingTransform(BatchInferenceTransform[Dict[str, Any], Dict[str, Any]]):
    """Batching wrapper for embedding generation with real batching logic."""
    
    def __init__(
        self,
        embedding_transform: EmbeddingTransform,
        application_store,
        provider_name: str,
        model_name: str,
        batch_size: int = 50,
        name: Optional[str] = None
    ):
        """Initialize batching embedding transform.
        
        Args:
            embedding_transform: The underlying embedder
            application_store: Storage for results
            provider_name: Embedding provider name
            model_name: Model name
            batch_size: Number of items per batch
            name: Optional component name
        """
        # Create adapters
        provider = EmbeddingBatchProvider(embedding_transform)
        store = EmbeddingBatchStore(application_store, provider_name, model_name)
        
        # Create config with hash function
        config = BatchConfig(
            batch_size=batch_size,
            hash_function=lambda item: create_embedding_task_hash(
                item, provider_name, model_name
            )
        )
        
        # Initialize parent
        super().__init__(
            provider=provider,
            store=store,
            config=config,
            name=name or "BatchEmbeddingTransform"
        )
        
        # Store references for introspection
        self.embedding_transform = embedding_transform
        self.provider_name = provider_name
        self.model_name = model_name
        self.application_store = application_store
    
    def process(self, inputs: List[Dict[str, Any]]) -> Any:
        """Override process to pass item map to store for hash verification."""
        # Create a map of hash keys to items for the store to use
        item_map = {}
        for item in inputs:
            hash_key = self.batch_config.hash_function(item)
            item_map[hash_key] = item
        
        # Temporarily attach to store for hash verification
        self.store._current_item_map = item_map
        
        try:
            # Call parent process method
            return super().process(inputs)
        finally:
            # Clean up
            if hasattr(self.store, '_current_item_map'):
                delattr(self.store, '_current_item_map')
    
    @property
    def input_types(self) -> List[type]:
        """Return supported input types."""
        return [List[dict]]  # List of summary dictionaries
    
    @property
    def output_types(self) -> List[type]:
        """Return output types."""
        return [List[dict]]  # List of embedding dictionaries