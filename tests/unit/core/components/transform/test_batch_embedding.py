"""Unit tests for BatchEmbeddingTransform."""

import pytest
from typing import List
from unittest.mock import Mock, MagicMock, patch, call

from apiana.core.components.transform.batch_embedding import (
    BatchEmbeddingTransform,
    EmbeddingBatchProvider,
    EmbeddingBatchStore,
    create_embedding_task_hash
)
from apiana.core.components.transform.embedder import EmbeddingTransform
from apiana.core.components.common import ComponentResult


class TestEmbeddingBatchProvider:
    """Tests for EmbeddingBatchProvider."""
    
    def test_process_batch_success(self):
        """Test successful batch processing."""
        # Create mock embedder
        embedder = Mock(spec=EmbeddingTransform)
        embeddings = [
            {
                "fragment_id": "1",
                "summary": "Summary 1",
                "embedding": [0.1, 0.2, 0.3],
                "embedding_dim": 3
            },
            {
                "fragment_id": "2",
                "summary": "Summary 2",
                "embedding": [0.4, 0.5, 0.6],
                "embedding_dim": 3
            }
        ]
        embedder.transform.return_value = ComponentResult(
            data=embeddings,
            metadata={},
            errors=[],
            execution_time_ms=100
        )
        
        # Create provider
        provider = EmbeddingBatchProvider(embedder)
        
        # Process batch
        summaries = [
            {"fragment_id": "1", "summary": "Summary 1"},
            {"fragment_id": "2", "summary": "Summary 2"}
        ]
        result = provider.process_batch(summaries)
        
        # Verify
        assert result == embeddings
        embedder.transform.assert_called_once_with(summaries)
    
    def test_process_batch_with_errors(self):
        """Test batch processing with errors."""
        embedder = Mock(spec=EmbeddingTransform)
        embeddings = [
            {
                "fragment_id": "1",
                "summary": "Summary 1",
                "embedding": [0.1, 0.2, 0.3],
                "embedding_dim": 3
            },
            {
                "fragment_id": "2",
                "summary": "Summary 2",
                "embedding": None,
                "error": "API error"
            }
        ]
        embedder.transform.return_value = ComponentResult(
            data=embeddings,
            metadata={},
            errors=["Failed to generate embedding for item 1"],
            execution_time_ms=100
        )
        
        provider = EmbeddingBatchProvider(embedder)
        summaries = [
            {"fragment_id": "1", "summary": "Summary 1"},
            {"fragment_id": "2", "summary": "Summary 2"}
        ]
        
        result = provider.process_batch(summaries)
        
        assert len(result) == 2
        assert result[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result[1]["embedding"] is None
        assert result[1]["error"] == "API error"


class TestEmbeddingBatchStore:
    """Tests for EmbeddingBatchStore."""
    
    def test_get_by_hash_found(self):
        """Test cache lookup when embedding exists."""
        mock_store = Mock()
        mock_store.get_embedding.return_value = {
            "vector": [0.1, 0.2, 0.3],
            "model_dimensions": 3,
            "embedding_id": "frag-123:hash456"
        }
        
        store = EmbeddingBatchStore(mock_store, "openai", "text-embedding-ada-002")
        result = store.get_by_hash("frag-123:hash456")
        
        assert result is not None
        assert result["fragment_id"] == "frag-123"
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["embedding_dim"] == 3
        
        mock_store.get_embedding.assert_called_once_with(
            source_id="frag-123",
            model_provider="openai",
            model_name="text-embedding-ada-002",
            source_text=None
        )
    
    def test_get_by_hash_not_found(self):
        """Test cache lookup when not found."""
        mock_store = Mock()
        mock_store.get_embedding.return_value = None
        
        store = EmbeddingBatchStore(mock_store, "openai", "text-embedding-ada-002")
        result = store.get_by_hash("frag-123:hash456")
        
        assert result is None
    
    def test_store_result_success(self):
        """Test storing embedding result."""
        mock_store = Mock()
        store = EmbeddingBatchStore(mock_store, "openai", "text-embedding-ada-002")
        
        input_item = {
            "fragment_id": "frag-123",
            "summary": "A test summary",
            "title": "Test",
            "original_messages": 5
        }
        
        result = {
            "fragment_id": "frag-123",
            "summary": "A test summary",
            "embedding": [0.1, 0.2, 0.3],
            "embedding_dim": 3
        }
        
        store.store_result("frag-123:hash456", input_item, result)
        
        mock_store.store_embedding.assert_called_once_with(
            source_id="frag-123",
            source_type="summary",
            source_text="A test summary",
            embedding_vector=[0.1, 0.2, 0.3],
            model_provider="openai",
            model_name="text-embedding-ada-002",
            metadata={
                "title": "Test",
                "original_messages": 5
            }
        )
    
    def test_store_result_no_embedding(self):
        """Test storing result without embedding doesn't store anything."""
        mock_store = Mock()
        store = EmbeddingBatchStore(mock_store, "openai", "text-embedding-ada-002")
        
        input_item = {"fragment_id": "frag-123", "summary": "Test"}
        result = {"fragment_id": "frag-123", "embedding": None, "error": "Failed"}
        
        store.store_result("frag-123:hash456", input_item, result)
        
        mock_store.store_embedding.assert_not_called()
    
    def test_store_result_no_summary(self):
        """Test storing result without summary doesn't store anything."""
        mock_store = Mock()
        store = EmbeddingBatchStore(mock_store, "openai", "text-embedding-ada-002")
        
        input_item = {"fragment_id": "frag-123", "summary": None}
        result = {"fragment_id": "frag-123", "embedding": [0.1, 0.2]}
        
        store.store_result("frag-123:hash456", input_item, result)
        
        mock_store.store_embedding.assert_not_called()


class TestCreateEmbeddingTaskHash:
    """Tests for create_embedding_task_hash function."""
    
    @patch('apiana.core.components.transform.batch_embedding.create_embedding_hash')
    def test_hash_creation(self, mock_create_hash):
        """Test hash is created correctly."""
        mock_create_hash.return_value = "text_hash_123"
        
        summary_item = {
            "fragment_id": "frag-456",
            "summary": "This is a test summary"
        }
        
        hash_result = create_embedding_task_hash(summary_item, "openai", "ada-002")
        
        assert hash_result == "frag-456:text_hash_123"
        mock_create_hash.assert_called_once_with(
            "This is a test summary", "openai", "ada-002"
        )
    
    def test_hash_missing_fields(self):
        """Test hash creation with missing fields."""
        summary_item = {}
        
        hash_result = create_embedding_task_hash(summary_item, "openai", "ada-002")
        
        # Should still create a hash with defaults
        assert hash_result.startswith("unknown:")
    
    @patch('apiana.core.components.transform.batch_embedding.create_embedding_hash')
    def test_hash_deterministic(self, mock_create_hash):
        """Test hash is deterministic."""
        mock_create_hash.return_value = "consistent_hash"
        
        summary_item = {
            "fragment_id": "frag-123",
            "summary": "Test summary"
        }
        
        hash1 = create_embedding_task_hash(summary_item, "openai", "ada")
        hash2 = create_embedding_task_hash(summary_item, "openai", "ada")
        
        assert hash1 == hash2
        assert hash1 == "frag-123:consistent_hash"


class TestBatchEmbeddingTransform:
    """Tests for the main batching embedding transform."""
    
    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedding transform."""
        embedder = Mock(spec=EmbeddingTransform)
        return embedder
    
    @pytest.fixture
    def mock_store(self):
        """Create mock application store."""
        return Mock()
    
    def test_initialization(self, mock_embedder, mock_store):
        """Test transform initialization."""
        transform = BatchEmbeddingTransform(
            embedding_transform=mock_embedder,
            application_store=mock_store,
            provider_name="openai",
            model_name="text-embedding-ada-002",
            batch_size=20
        )
        
        assert transform.embedding_transform == mock_embedder
        assert transform.provider_name == "openai"
        assert transform.model_name == "text-embedding-ada-002"
        assert transform.batch_config.batch_size == 20
        assert transform.name == "BatchEmbeddingTransform"
    
    def test_input_output_types(self, mock_embedder, mock_store):
        """Test type specifications."""
        transform = BatchEmbeddingTransform(
            mock_embedder, mock_store, "openai", "ada-002"
        )
        
        assert transform.input_types == [List[dict]]
        assert transform.output_types == [List[dict]]
    
    def test_default_batch_size(self, mock_embedder, mock_store):
        """Test default batch size is 50."""
        transform = BatchEmbeddingTransform(
            mock_embedder, mock_store, "openai", "ada-002"
        )
        
        assert transform.batch_config.batch_size == 50
    
    def test_process_with_real_batching(self, mock_embedder, mock_store):
        """Test real batching behavior."""
        # Create summary items
        summaries = [
            {"fragment_id": f"frag-{i}", "summary": f"Summary text {i}", "title": f"Title {i}"}
            for i in range(7)
        ]
        
        # Mock store - no embeddings exist yet
        mock_store.get_embedding.return_value = None
        
        # Track what gets embedded
        embedded_items = []
        
        def mock_transform(items):
            embedded_items.extend(items)
            return ComponentResult(
                data=[
                    {
                        **item,
                        "embedding": [float(i) * 0.1 for i in range(10)],
                        "embedding_dim": 10
                    }
                    for item in items
                ],
                metadata={"items_processed": len(items)},
                errors=[],
                execution_time_ms=50
            )
        
        mock_embedder.transform.side_effect = mock_transform
        
        # Create transform with batch size of 3
        transform = BatchEmbeddingTransform(
            mock_embedder, mock_store, "openai", "text-embedding-ada-002", batch_size=3
        )
        
        # Process summaries
        result = transform.process(summaries)
        
        # Verify batching occurred
        assert mock_embedder.transform.call_count == 3  # 7 items / batch_size 3 = 3 calls
        assert len(embedded_items) == 7
        assert len(result.data) == 7
        
        # Verify all items got embeddings
        for i, embedding_result in enumerate(result.data):
            assert embedding_result["fragment_id"] == f"frag-{i}"
            assert embedding_result["summary"] == f"Summary text {i}"
            assert len(embedding_result["embedding"]) == 10
            assert embedding_result["embedding_dim"] == 10
    
    def test_process_with_cached_embeddings(self, mock_embedder, mock_store):
        """Test that cached embeddings are not re-processed."""
        # Create summary items
        summaries = [
            {"fragment_id": f"frag-{i}", "summary": f"Summary text {i}", "title": f"Title {i}"}
            for i in range(3)
        ]
        
        # Mock store - first embedding is cached
        def mock_get_embedding(source_id, model_provider, model_name, source_text=None):
            if source_id == "frag-0":
                return {
                    "vector": [0.1, 0.2, 0.3],
                    "model_dimensions": 3,
                    "embedding_id": "frag-0:hash123"
                }
            return None
        
        mock_store.get_embedding.side_effect = mock_get_embedding
        
        # Only uncached items should be embedded
        mock_embedder.transform.return_value = ComponentResult(
            data=[
                {
                    "fragment_id": "frag-1",
                    "summary": "Summary text 1",
                    "title": "Title 1",
                    "embedding": [0.4, 0.5, 0.6],
                    "embedding_dim": 3
                },
                {
                    "fragment_id": "frag-2",
                    "summary": "Summary text 2",
                    "title": "Title 2",
                    "embedding": [0.7, 0.8, 0.9],
                    "embedding_dim": 3
                }
            ],
            metadata={},
            errors=[],
            execution_time_ms=50
        )
        
        # Create transform
        transform = BatchEmbeddingTransform(
            mock_embedder, mock_store, "openai", "text-embedding-ada-002", batch_size=10
        )
        
        # Process summaries
        result = transform.process(summaries)
        
        # Verify only uncached items were processed
        mock_embedder.transform.assert_called_once()
        call_args = mock_embedder.transform.call_args[0][0]
        assert len(call_args) == 2  # Only frag-1 and frag-2
        assert call_args[0]["fragment_id"] == "frag-1"
        assert call_args[1]["fragment_id"] == "frag-2"
        
        # Result should include all items in original order
        assert len(result.data) == 3
        assert result.data[0]["fragment_id"] == "frag-0"
        assert result.data[0]["_cached"] == True
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        
        assert result.data[1]["fragment_id"] == "frag-1"
        assert "_cached" not in result.data[1]
        assert result.data[1]["embedding"] == [0.4, 0.5, 0.6]
        
        assert result.data[2]["fragment_id"] == "frag-2"
        assert "_cached" not in result.data[2]
        assert result.data[2]["embedding"] == [0.7, 0.8, 0.9]
    
    def test_custom_name(self, mock_embedder, mock_store):
        """Test custom component name."""
        transform = BatchEmbeddingTransform(
            mock_embedder, mock_store, "cohere", "embed-v3",
            name="CustomEmbeddingBatcher"
        )
        
        assert transform.name == "CustomEmbeddingBatcher"