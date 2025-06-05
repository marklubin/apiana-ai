"""Adapter to make ApplicationStore compatible with BatchStore protocol."""

import json
import hashlib
from typing import Optional, Dict, Any

from apiana.core.components.transform.batch_inference import BatchStore
from apiana.stores.neo4j.application_store import ApplicationStore
from apiana.types.chat_fragment import ChatFragment


class ChatFragmentStoreAdapter(BatchStore[ChatFragment, Dict[str, Any]]):
    """
    Adapter that makes ApplicationStore compatible with the BatchStore protocol
    for ChatFragment batch processing.
    
    This adapter:
    - Uses a composite key of fragment content + model provider + model name
    - Stores processing results as metadata on the ChatFragment
    - Retrieves cached results from fragment metadata
    """
    
    def __init__(
        self,
        application_store: ApplicationStore,
        metadata_key: str = "batch_inference_cache"
    ):
        """
        Initialize the adapter.
        
        Args:
            application_store: The underlying ApplicationStore instance
            metadata_key: Key under which to store batch inference results in metadata
        """
        self.store = application_store
        self.metadata_key = metadata_key
    
    def get_by_hash(self, hash_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result by hash key.
        
        The hash key format is expected to be:
        "{fragment_id}:{content_hash}:{provider}:{model}"
        
        Args:
            hash_key: The composite hash key
            
        Returns:
            Cached result if found, None otherwise
        """
        # Parse the hash key
        try:
            parts = hash_key.split(":", 3)
            if len(parts) != 4:
                return None
                
            fragment_id, content_hash, provider, model = parts
            
            # Get the fragment
            fragment = self.store.get_fragment(fragment_id)
            if not fragment:
                return None
            
            # Check if we have cached results in metadata
            if hasattr(fragment, 'message_metadata') and fragment.message_metadata:
                # Look for cached results in the last metadata entry
                last_metadata = fragment.message_metadata[-1] if fragment.message_metadata else {}
                
                if isinstance(last_metadata, dict) and self.metadata_key in last_metadata:
                    cache = last_metadata[self.metadata_key]
                    
                    # Verify this is the right cached result
                    if (cache.get('content_hash') == content_hash and
                        cache.get('provider') == provider and
                        cache.get('model') == model):
                        return cache.get('result')
            
            return None
            
        except Exception:
            # If anything goes wrong parsing the key, return None
            return None
    
    def store_result(
        self, 
        hash_key: str, 
        input_item: ChatFragment, 
        result: Dict[str, Any]
    ) -> None:
        """
        Store computation result with hash key.
        
        Args:
            hash_key: The composite hash key
            input_item: The original ChatFragment
            result: The processing result to cache
        """
        try:
            # Parse the hash key
            parts = hash_key.split(":", 3)
            if len(parts) != 4:
                return
                
            fragment_id, content_hash, provider, model = parts
            
            # Create cache entry
            cache_entry = {
                'content_hash': content_hash,
                'provider': provider,
                'model': model,
                'result': result,
                'cached_at': input_item.update_time.isoformat()
            }
            
            # Update fragment metadata with cached result
            if not hasattr(input_item, 'message_metadata') or not input_item.message_metadata:
                input_item.message_metadata = []
            
            # Add cache to the last metadata entry or create new one
            if input_item.message_metadata:
                last_metadata = input_item.message_metadata[-1]
                if isinstance(last_metadata, dict):
                    last_metadata[self.metadata_key] = cache_entry
                else:
                    input_item.message_metadata.append({self.metadata_key: cache_entry})
            else:
                input_item.message_metadata.append({self.metadata_key: cache_entry})
            
            # Store the updated fragment
            self.store.store_fragment(input_item, tags=['batch_inference_cached'])
            
        except Exception:
            # If storage fails, we continue without caching
            # This follows the fail-open pattern for caching
            pass


def create_chat_fragment_hash(
    fragment: ChatFragment,
    provider: str,
    model: str,
    include_metadata: bool = False
) -> str:
    """
    Create a hash key for a ChatFragment combined with model information.
    
    This creates a deterministic hash based on:
    - The fragment's message content
    - The provider name (e.g., "openai", "anthropic")
    - The model name (e.g., "gpt-4", "claude-3")
    
    Args:
        fragment: The ChatFragment to hash
        provider: The model provider name
        model: The model name
        include_metadata: Whether to include message metadata in the hash
        
    Returns:
        A composite hash key in format: "{fragment_id}:{content_hash}:{provider}:{model}"
    """
    # Create a stable hash of the message content
    hasher = hashlib.sha256()
    
    # Hash each message in order
    for i, msg in enumerate(fragment.messages):
        # Include role and content
        hasher.update(msg.get('role', '').encode('utf-8'))
        hasher.update(msg.get('content', '').encode('utf-8'))
        
        # Optionally include metadata
        if include_metadata and i < len(fragment.message_metadata):
            metadata = fragment.message_metadata[i]
            # Sort keys for deterministic ordering
            metadata_str = json.dumps(metadata, sort_keys=True)
            hasher.update(metadata_str.encode('utf-8'))
    
    # Add provider and model to ensure different models get different hashes
    hasher.update(provider.lower().encode('utf-8'))
    hasher.update(model.lower().encode('utf-8'))
    
    content_hash = hasher.hexdigest()[:16]  # Use first 16 chars for brevity
    
    # Return composite key
    return f"{fragment.fragment_id}:{content_hash}:{provider.lower()}:{model.lower()}"