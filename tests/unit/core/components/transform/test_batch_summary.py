"""Unit tests for BatchingChatFragmentSummaryTransform."""

import pytest
from typing import List
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime

from apiana.core.components.transform.batch_summary import (
    BatchingChatFragmentSummaryTransform,
    SummaryBatchProvider,
    SummaryBatchStore,
    create_summary_hash
)
from apiana.core.components.transform.summarizer import SummarizerTransform
from apiana.types.chat_fragment import ChatFragment
from apiana.core.components.common import ComponentResult


class TestSummaryBatchProvider:
    """Tests for SummaryBatchProvider."""
    
    def test_process_batch_success(self):
        """Test successful batch processing."""
        # Create mock summarizer
        summarizer = Mock(spec=SummarizerTransform)
        summaries = [
            {"fragment_id": "1", "summary": "Summary 1"},
            {"fragment_id": "2", "summary": "Summary 2"}
        ]
        summarizer.transform.return_value = ComponentResult(
            data=summaries,
            metadata={},
            errors=[],
            execution_time_ms=100
        )
        
        # Create provider
        provider = SummaryBatchProvider(summarizer)
        
        # Process batch
        fragments = [
            ChatFragment(fragment_id="1", messages=[]),
            ChatFragment(fragment_id="2", messages=[])
        ]
        result = provider.process_batch(fragments)
        
        # Verify
        assert result == summaries
        summarizer.transform.assert_called_once_with(fragments)
    
    def test_process_batch_with_errors(self):
        """Test batch processing with errors."""
        summarizer = Mock(spec=SummarizerTransform)
        summaries = [
            {"fragment_id": "1", "summary": "Summary 1"},
            {"fragment_id": "2", "summary": None, "error": "Failed"}
        ]
        summarizer.transform.return_value = ComponentResult(
            data=summaries,
            metadata={},
            errors=["Failed to summarize fragment 2"],
            execution_time_ms=100
        )
        
        provider = SummaryBatchProvider(summarizer)
        fragments = [
            ChatFragment(fragment_id="1", messages=[]),
            ChatFragment(fragment_id="2", messages=[])
        ]
        
        result = provider.process_batch(fragments)
        
        assert len(result) == 2
        assert result[0]["summary"] == "Summary 1"
        assert result[1]["summary"] is None
        assert result[1]["error"] == "Failed"


class TestSummaryBatchStore:
    """Tests for SummaryBatchStore."""
    
    def test_get_by_hash_not_found(self):
        """Test cache lookup when not found."""
        mock_store = Mock()
        mock_store.get_fragment.return_value = None
        
        store = SummaryBatchStore(mock_store, "openai", "gpt-4")
        result = store.get_by_hash("frag-123:hash456:openai:gpt-4")
        
        assert result is None
        mock_store.get_fragment.assert_called_once_with("frag-123")
    
    def test_store_result_success(self):
        """Test storing summary result."""
        mock_store = Mock()
        store = SummaryBatchStore(mock_store, "openai", "gpt-4")
        
        fragment = ChatFragment(
            fragment_id="frag-123",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        result = {
            "fragment_id": "frag-123",
            "summary": "A greeting conversation",
            "title": "Greeting"
        }
        
        store.store_result("frag-123:hash456", fragment, result)
        
        mock_store.store_fragment.assert_called_once_with(
            fragment,
            summary_agent_id="openai:gpt-4",
            summary_text="A greeting conversation"
        )
    
    def test_store_result_no_summary(self):
        """Test storing result without summary doesn't update fragment."""
        mock_store = Mock()
        store = SummaryBatchStore(mock_store, "openai", "gpt-4")
        
        fragment = ChatFragment(fragment_id="frag-123", messages=[])
        result = {"fragment_id": "frag-123", "summary": None, "error": "Failed"}
        
        store.store_result("frag-123:hash456", fragment, result)
        
        mock_store.store_fragment.assert_not_called()


class TestCreateSummaryHash:
    """Tests for create_summary_hash function."""
    
    def test_hash_creation(self):
        """Test hash is created correctly."""
        fragment = ChatFragment(
            fragment_id="frag-123",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            title="Test",
            create_time=datetime(2024, 1, 1)
        )
        
        hash1 = create_summary_hash(fragment, "openai", "gpt-4")
        hash2 = create_summary_hash(fragment, "openai", "gpt-4")
        
        # Should be deterministic
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0
    
    def test_hash_different_for_different_inputs(self):
        """Test hash differs for different inputs."""
        fragment1 = ChatFragment(
            fragment_id="frag-1",
            messages=[{"role": "user", "content": "Hello"}]
        )
        fragment2 = ChatFragment(
            fragment_id="frag-2",
            messages=[{"role": "user", "content": "Goodbye"}]
        )
        
        hash1 = create_summary_hash(fragment1, "openai", "gpt-4")
        hash2 = create_summary_hash(fragment2, "openai", "gpt-4")
        
        assert hash1 != hash2


class TestBatchingChatFragmentSummaryTransform:
    """Tests for the main batching summary transform."""
    
    @pytest.fixture
    def mock_summarizer(self):
        """Create mock summarizer transform."""
        summarizer = Mock(spec=SummarizerTransform)
        return summarizer
    
    @pytest.fixture
    def mock_store(self):
        """Create mock application store."""
        return Mock()
    
    def test_initialization(self, mock_summarizer, mock_store):
        """Test transform initialization."""
        transform = BatchingChatFragmentSummaryTransform(
            summarizer_transform=mock_summarizer,
            application_store=mock_store,
            provider_name="openai",
            model_name="gpt-4",
            batch_size=5
        )
        
        assert transform.summarizer_transform == mock_summarizer
        assert transform.provider_name == "openai"
        assert transform.model_name == "gpt-4"
        assert transform.batch_config.batch_size == 5
        assert transform.name == "BatchingChatFragmentSummaryTransform"
    
    def test_input_output_types(self, mock_summarizer, mock_store):
        """Test type specifications."""
        transform = BatchingChatFragmentSummaryTransform(
            mock_summarizer, mock_store, "openai", "gpt-4"
        )
        
        assert transform.input_types == [List[ChatFragment]]
        assert transform.output_types == [List[dict]]
    
    def test_process_with_real_batching(self, mock_summarizer, mock_store):
        """Test real batching behavior with cache hits and misses."""
        # Create fragments
        fragments = [
            ChatFragment(fragment_id=f"frag-{i}", messages=[{"role": "user", "content": f"Message {i}"}])
            for i in range(5)
        ]
        
        # Mock store behavior
        mock_store.ChatFragmentNode = Mock()
        mock_store.ChatFragmentNode.nodes.filter.return_value.first.return_value = None
        mock_store.get_fragment.return_value = None
        
        # Track what gets summarized
        summarized_fragments = []
        
        def mock_transform(frags):
            summarized_fragments.extend(frags)
            return ComponentResult(
                data=[
                    {
                        "fragment_id": f.fragment_id,
                        "title": f.title,
                        "summary": f"Summary of {f.fragment_id}",
                        "original_messages": len(f.messages)
                    }
                    for f in frags
                ],
                metadata={"fragments_processed": len(frags)},
                errors=[],
                execution_time_ms=100
            )
        
        mock_summarizer.transform.side_effect = mock_transform
        
        # Create transform with batch size of 2
        transform = BatchingChatFragmentSummaryTransform(
            mock_summarizer, mock_store, "openai", "gpt-4", batch_size=2
        )
        
        # Process fragments
        result = transform.process(fragments)
        
        # Verify batching occurred
        assert mock_summarizer.transform.call_count == 3  # 5 items / batch_size 2 = 3 calls
        assert len(summarized_fragments) == 5
        assert len(result.data) == 5
        
        # Verify all fragments got summaries
        for i, summary in enumerate(result.data):
            assert summary["fragment_id"] == f"frag-{i}"
            assert summary["summary"] == f"Summary of frag-{i}"
    
    @patch('apiana.core.components.transform.batch_summary.create_summary_hash')
    def test_process_with_cached_items(self, mock_hash, mock_summarizer, mock_store):
        """Test that cached items are not re-processed."""
        # Create fragments
        fragments = [
            ChatFragment(fragment_id=f"frag-{i}", messages=[{"role": "user", "content": f"Message {i}"}])
            for i in range(3)
        ]
        
        # Setup hash function
        mock_hash.side_effect = lambda f, p, m: f"{f.fragment_id}:hash:{p}:{m}"
        
        # Create transform first to setup the store
        transform = BatchingChatFragmentSummaryTransform(
            mock_summarizer, mock_store, "openai", "gpt-4", batch_size=10
        )
        
        # Mock store - simulate first fragment is already summarized
        mock_store.ChatFragmentNode = Mock()
        mock_store.ChatFragmentNode.nodes.filter.return_value.first.return_value = None
        
        # Pre-populate the store's cache with the first item
        cached_result = {
            "fragment_id": "frag-0",
            "title": "Fragment 0",
            "summary": "CACHED_SUMMARY",
            "original_messages": 1,
            "_cached": True
        }
        transform.store.summary_cache["frag-0:hash:openai:gpt-4"] = cached_result
        
        # Mock get_fragment to return fragment for cached check
        def mock_get_fragment(frag_id):
            if frag_id == "frag-0":
                return Mock(
                    fragment_id="frag-0",
                    title="Fragment 0",
                    messages=[{"role": "user", "content": "Message 0"}]
                )
            return None
        
        mock_store.get_fragment.side_effect = mock_get_fragment
        
        # Setup the filter mock to return the cached summary
        def mock_filter_first(fragment_id=None):
            if fragment_id == "frag-0":
                node = Mock()
                node.summary_agent_id = "openai:gpt-4"
                return node
            return None
        
        mock_filter = Mock()
        mock_filter.first.side_effect = mock_filter_first
        mock_store.ChatFragmentNode.nodes.filter.return_value = mock_filter
        
        # Mock summarizer should only be called with uncached items
        def mock_transform(frags):
            # Should only receive uncached fragments
            return ComponentResult(
                data=[
                    {"fragment_id": f.fragment_id, "summary": f"Summary {f.fragment_id[-1]}", 
                     "title": f.title, "original_messages": len(f.messages)}
                    for f in frags
                ],
                metadata={},
                errors=[],
                execution_time_ms=100
            )
        
        mock_summarizer.transform.side_effect = mock_transform
        
        # Process fragments
        result = transform.process(fragments)
        
        # Verify only uncached items were processed
        mock_summarizer.transform.assert_called_once()
        call_args = mock_summarizer.transform.call_args[0][0]
        assert len(call_args) == 2  # Only frag-1 and frag-2
        assert call_args[0].fragment_id == "frag-1"
        assert call_args[1].fragment_id == "frag-2"
        
        # Result should include all items
        assert len(result.data) == 3
        assert result.data[0]["fragment_id"] == "frag-0"
        assert result.data[0]["_cached"] == True  # First item was cached
        assert result.data[1]["fragment_id"] == "frag-1"
        assert "_cached" not in result.data[1]  # Others were processed
        assert result.data[2]["fragment_id"] == "frag-2"
        assert "_cached" not in result.data[2]
    
    def test_custom_name(self, mock_summarizer, mock_store):
        """Test custom component name."""
        transform = BatchingChatFragmentSummaryTransform(
            mock_summarizer, mock_store, "openai", "gpt-4",
            name="CustomSummaryBatcher"
        )
        
        assert transform.name == "CustomSummaryBatcher"