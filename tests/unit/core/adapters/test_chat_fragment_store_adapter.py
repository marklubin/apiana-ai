"""Unit tests for ChatFragmentStoreAdapter and hash function."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from apiana.core.adapters.chat_fragment_store_adapter import (
    ChatFragmentStoreAdapter,
    create_chat_fragment_hash
)
from apiana.types.chat_fragment import ChatFragment


class TestCreateChatFragmentHash:
    """Test cases for the create_chat_fragment_hash function."""
    
    def test_basic_hash_creation(self):
        """Test basic hash creation with minimal fragment."""
        fragment = ChatFragment(
            fragment_id="test-123",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        )
        
        hash_key = create_chat_fragment_hash(fragment, "openai", "gpt-4")
        
        # Check format
        parts = hash_key.split(":")
        assert len(parts) == 4
        assert parts[0] == "test-123"
        assert len(parts[1]) == 16  # Content hash length
        assert parts[2] == "openai"
        assert parts[3] == "gpt-4"
    
    def test_hash_deterministic(self):
        """Test that same inputs produce same hash."""
        fragment = ChatFragment(
            fragment_id="test-123",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        )
        
        hash1 = create_chat_fragment_hash(fragment, "openai", "gpt-4")
        hash2 = create_chat_fragment_hash(fragment, "openai", "gpt-4")
        
        assert hash1 == hash2
    
    def test_hash_different_content(self):
        """Test that different content produces different hashes."""
        fragment1 = ChatFragment(
            fragment_id="test-123",
            messages=[{"role": "user", "content": "Hello"}]
        )
        fragment2 = ChatFragment(
            fragment_id="test-123",
            messages=[{"role": "user", "content": "Goodbye"}]
        )
        
        hash1 = create_chat_fragment_hash(fragment1, "openai", "gpt-4")
        hash2 = create_chat_fragment_hash(fragment2, "openai", "gpt-4")
        
        # Fragment IDs are same but content hashes should differ
        parts1 = hash1.split(":")
        parts2 = hash2.split(":")
        assert parts1[0] == parts2[0]  # Same fragment ID
        assert parts1[1] != parts2[1]  # Different content hash
    
    def test_hash_different_provider(self):
        """Test that different providers produce different hashes."""
        fragment = ChatFragment(
            fragment_id="test-123",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        hash1 = create_chat_fragment_hash(fragment, "openai", "gpt-4")
        hash2 = create_chat_fragment_hash(fragment, "anthropic", "gpt-4")
        
        assert hash1 != hash2
        assert "openai" in hash1
        assert "anthropic" in hash2
    
    def test_hash_different_model(self):
        """Test that different models produce different hashes."""
        fragment = ChatFragment(
            fragment_id="test-123",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        hash1 = create_chat_fragment_hash(fragment, "openai", "gpt-4")
        hash2 = create_chat_fragment_hash(fragment, "openai", "gpt-3.5-turbo")
        
        assert hash1 != hash2
        assert "gpt-4" in hash1
        assert "gpt-3.5-turbo" in hash2
    
    def test_hash_case_insensitive(self):
        """Test that provider and model names are case-insensitive."""
        fragment = ChatFragment(
            fragment_id="test-123",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        hash1 = create_chat_fragment_hash(fragment, "OpenAI", "GPT-4")
        hash2 = create_chat_fragment_hash(fragment, "openai", "gpt-4")
        
        assert hash1 == hash2
    
    def test_hash_with_metadata(self):
        """Test hash creation with metadata included."""
        fragment = ChatFragment(
            fragment_id="test-123",
            messages=[{"role": "user", "content": "Hello"}],
            message_metadata=[{"timestamp": "2024-01-01T00:00:00"}]
        )
        
        hash_without = create_chat_fragment_hash(fragment, "openai", "gpt-4", include_metadata=False)
        hash_with = create_chat_fragment_hash(fragment, "openai", "gpt-4", include_metadata=True)
        
        # Hashes should be different when metadata is included
        assert hash_without != hash_with
    
    def test_hash_empty_messages(self):
        """Test hash creation with empty messages."""
        fragment = ChatFragment(
            fragment_id="test-123",
            messages=[]
        )
        
        hash_key = create_chat_fragment_hash(fragment, "openai", "gpt-4")
        
        # Should still produce valid hash
        parts = hash_key.split(":")
        assert len(parts) == 4
        assert parts[0] == "test-123"
    
    def test_hash_message_order_matters(self):
        """Test that message order affects the hash."""
        fragment1 = ChatFragment(
            fragment_id="test-123",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]
        )
        fragment2 = ChatFragment(
            fragment_id="test-123",
            messages=[
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Hello"}
            ]
        )
        
        hash1 = create_chat_fragment_hash(fragment1, "openai", "gpt-4")
        hash2 = create_chat_fragment_hash(fragment2, "openai", "gpt-4")
        
        # Different order should produce different hashes
        assert hash1 != hash2


class TestChatFragmentStoreAdapter:
    """Test cases for ChatFragmentStoreAdapter."""
    
    @pytest.fixture
    def mock_store(self):
        """Create a mock ApplicationStore."""
        return Mock()
    
    @pytest.fixture
    def adapter(self, mock_store):
        """Create adapter instance."""
        return ChatFragmentStoreAdapter(mock_store)
    
    def test_adapter_initialization(self, mock_store):
        """Test adapter initialization."""
        adapter = ChatFragmentStoreAdapter(mock_store, metadata_key="custom_key")
        assert adapter.store == mock_store
        assert adapter.metadata_key == "custom_key"
    
    def test_get_by_hash_not_found(self, adapter, mock_store):
        """Test get_by_hash when fragment is not found."""
        mock_store.get_fragment.return_value = None
        
        result = adapter.get_by_hash("frag-123:abc123:openai:gpt-4")
        
        assert result is None
        mock_store.get_fragment.assert_called_once_with("frag-123")
    
    def test_get_by_hash_invalid_key_format(self, adapter):
        """Test get_by_hash with invalid key format."""
        result = adapter.get_by_hash("invalid-key")
        assert result is None
    
    def test_get_by_hash_no_cached_result(self, adapter, mock_store):
        """Test get_by_hash when fragment exists but has no cached result."""
        fragment = ChatFragment(
            fragment_id="frag-123",
            messages=[{"role": "user", "content": "Hello"}]
        )
        mock_store.get_fragment.return_value = fragment
        
        result = adapter.get_by_hash("frag-123:abc123:openai:gpt-4")
        
        assert result is None
    
    def test_get_by_hash_cached_result_found(self, adapter, mock_store):
        """Test get_by_hash when cached result exists."""
        cached_result = {"summary": "Test summary"}
        fragment = ChatFragment(
            fragment_id="frag-123",
            messages=[{"role": "user", "content": "Hello"}],
            message_metadata=[{
                "batch_inference_cache": {
                    "content_hash": "abc123",
                    "provider": "openai",
                    "model": "gpt-4",
                    "result": cached_result
                }
            }]
        )
        mock_store.get_fragment.return_value = fragment
        
        result = adapter.get_by_hash("frag-123:abc123:openai:gpt-4")
        
        assert result == cached_result
    
    def test_get_by_hash_wrong_provider(self, adapter, mock_store):
        """Test get_by_hash when cached result is for different provider."""
        fragment = ChatFragment(
            fragment_id="frag-123",
            messages=[{"role": "user", "content": "Hello"}],
            message_metadata=[{
                "batch_inference_cache": {
                    "content_hash": "abc123",
                    "provider": "anthropic",  # Different provider
                    "model": "gpt-4",
                    "result": {"summary": "Test"}
                }
            }]
        )
        mock_store.get_fragment.return_value = fragment
        
        result = adapter.get_by_hash("frag-123:abc123:openai:gpt-4")
        
        assert result is None
    
    def test_store_result_success(self, adapter, mock_store):
        """Test successful result storage."""
        fragment = ChatFragment(
            fragment_id="frag-123",
            messages=[{"role": "user", "content": "Hello"}],
            update_time=datetime(2024, 1, 1)
        )
        result = {"summary": "Test summary"}
        
        adapter.store_result("frag-123:abc123:openai:gpt-4", fragment, result)
        
        # Check that fragment was updated with cache
        assert len(fragment.message_metadata) == 1
        cache = fragment.message_metadata[0]["batch_inference_cache"]
        assert cache["content_hash"] == "abc123"
        assert cache["provider"] == "openai"
        assert cache["model"] == "gpt-4"
        assert cache["result"] == result
        
        # Check that store was called
        mock_store.store_fragment.assert_called_once_with(
            fragment,
            tags=['batch_inference_cached']
        )
    
    def test_store_result_existing_metadata(self, adapter, mock_store):
        """Test storing result when fragment already has metadata."""
        fragment = ChatFragment(
            fragment_id="frag-123",
            messages=[{"role": "user", "content": "Hello"}],
            message_metadata=[{"existing": "data"}],
            update_time=datetime(2024, 1, 1)
        )
        result = {"summary": "Test summary"}
        
        adapter.store_result("frag-123:abc123:openai:gpt-4", fragment, result)
        
        # Check that cache was added to existing metadata
        assert len(fragment.message_metadata) == 1
        assert fragment.message_metadata[0]["existing"] == "data"
        assert "batch_inference_cache" in fragment.message_metadata[0]
    
    def test_store_result_invalid_key(self, adapter, mock_store):
        """Test store_result with invalid key format."""
        fragment = ChatFragment(fragment_id="frag-123")
        
        # Should not raise exception
        adapter.store_result("invalid-key", fragment, {"result": "data"})
        
        # Store should not be called
        mock_store.store_fragment.assert_not_called()
    
    def test_store_result_storage_failure(self, adapter, mock_store):
        """Test that storage failures are handled gracefully."""
        fragment = ChatFragment(
            fragment_id="frag-123",
            update_time=datetime(2024, 1, 1)
        )
        mock_store.store_fragment.side_effect = Exception("Storage error")
        
        # Should not raise exception (fail-open pattern)
        adapter.store_result("frag-123:abc123:openai:gpt-4", fragment, {"result": "data"})
    
    def test_custom_metadata_key(self, mock_store):
        """Test using custom metadata key."""
        adapter = ChatFragmentStoreAdapter(mock_store, metadata_key="my_cache")
        
        fragment = ChatFragment(
            fragment_id="frag-123",
            messages=[{"role": "user", "content": "Hello"}],
            message_metadata=[{
                "my_cache": {
                    "content_hash": "abc123",
                    "provider": "openai",
                    "model": "gpt-4",
                    "result": {"data": "test"}
                }
            }]
        )
        mock_store.get_fragment.return_value = fragment
        
        result = adapter.get_by_hash("frag-123:abc123:openai:gpt-4")
        
        assert result == {"data": "test"}