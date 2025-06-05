"""Unit tests for batch inference transform component."""

import pytest
from typing import List, Optional
from unittest.mock import Mock

from apiana.core.components.transform.batch_inference import (
    BatchInferenceTransform,
    BatchConfig,
)
from apiana.core.components.common.base import ComponentResult


class MockProvider:
    """Mock provider for testing."""
    
    def __init__(self, process_func=None, fail_count=0):
        self.process_func = process_func
        self.fail_count = fail_count
        self.call_count = 0
        self.calls = []
    
    def process_batch(self, items: List[str]) -> List[str]:
        """Process a batch of items."""
        self.calls.append(items)
        self.call_count += 1
        
        if self.call_count <= self.fail_count:
            raise Exception(f"Mock failure {self.call_count}")
        
        if self.process_func:
            return self.process_func(items)
        
        # Default: uppercase each item
        return [item.upper() for item in items]


class MockStore:
    """Mock store for testing."""
    
    def __init__(self):
        self.storage = {}
    
    def get_by_hash(self, hash_key: str) -> Optional[str]:
        """Get item by hash."""
        return self.storage.get(hash_key)
    
    def store_result(self, hash_key: str, input_item: str, result: str) -> None:
        """Store result."""
        self.storage[hash_key] = result
    


def simple_hash(item: str) -> str:
    """Simple hash function for testing."""
    return f"hash_{item}"


class TestBatchInferenceTransform:
    """Test cases for BatchInferenceTransform."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        return MockProvider()
    
    @pytest.fixture
    def mock_store(self):
        """Create mock store."""
        return MockStore()
    
    @pytest.fixture
    def config(self):
        """Create default config."""
        return BatchConfig(
            batch_size=3,
            hash_function=simple_hash,
            max_retries=3,
            retry_delay=0.1
        )
    
    @pytest.fixture
    def transform(self, mock_provider, mock_store, config):
        """Create transform instance."""
        return BatchInferenceTransform(
            provider=mock_provider,
            store=mock_store,
            config=config
        )
    
    def test_empty_input(self, transform):
        """Test processing empty input."""
        result = transform.process([])
        assert isinstance(result, ComponentResult)
        assert result.data == []
        assert result.success
    
    def test_single_batch(self, transform):
        """Test processing items that fit in single batch."""
        inputs = ["a", "b", "c"]
        result = transform.process(inputs)
        
        assert isinstance(result, ComponentResult)
        assert result.data == ["A", "B", "C"]
        assert result.success
        assert transform.provider.call_count == 1
        assert transform.provider.calls[0] == inputs
        assert result.metadata["total_items"] == 3
        assert result.metadata["batch_count"] == 1
    
    def test_multiple_batches(self, transform):
        """Test processing items that require multiple batches."""
        inputs = ["a", "b", "c", "d", "e", "f", "g"]
        result = transform.process(inputs)
        
        assert isinstance(result, ComponentResult)
        assert result.data == ["A", "B", "C", "D", "E", "F", "G"]
        assert result.success
        assert transform.provider.call_count == 3
        assert transform.provider.calls[0] == ["a", "b", "c"]
        assert transform.provider.calls[1] == ["d", "e", "f"]
        assert transform.provider.calls[2] == ["g"]
        assert result.metadata["total_items"] == 7
        assert result.metadata["batch_count"] == 3
    
    def test_exact_batch_boundaries(self, transform):
        """Test when input size is exact multiple of batch size."""
        inputs = ["a", "b", "c", "d", "e", "f"]
        result = transform.process(inputs)
        
        assert result.data == ["A", "B", "C", "D", "E", "F"]
        assert result.success
        assert transform.provider.call_count == 2
        assert transform.provider.calls[0] == ["a", "b", "c"]
        assert transform.provider.calls[1] == ["d", "e", "f"]
    
    def test_retry_on_failure(self, mock_store, config):
        """Test retry logic on provider failure."""
        # Provider fails once then succeeds
        provider = MockProvider(fail_count=1)
        transform = BatchInferenceTransform(
            provider=provider,
            store=mock_store,
            config=config
        )
        
        inputs = ["a", "b"]
        result = transform.process(inputs)
        
        assert result.data == ["A", "B"]
        assert result.success
        assert provider.call_count == 2  # First attempt failed, second succeeded
    
    def test_max_retries_exceeded(self, mock_store, config):
        """Test when all retries are exhausted."""
        # Provider always fails
        provider = MockProvider(fail_count=10)
        transform = BatchInferenceTransform(
            provider=provider,
            store=mock_store,
            config=config
        )
        
        inputs = ["a", "b"]
        with pytest.raises(RuntimeError) as exc_info:
            transform.process(inputs)
        
        assert "failed after 3 attempts" in str(exc_info.value)
        assert provider.call_count == 3
    
    def test_provider_wrong_result_count(self, mock_store, config):
        """Test when provider returns wrong number of results."""
        # Provider returns wrong number of results
        provider = MockProvider(process_func=lambda items: ["X"])
        transform = BatchInferenceTransform(
            provider=provider,
            store=mock_store,
            config=config
        )
        
        inputs = ["a", "b"]
        with pytest.raises(RuntimeError) as exc_info:
            transform.process(inputs)
        
        assert "Provider returned 1 results for 2 inputs" in str(exc_info.value)
    
    def test_large_batch_processing(self, transform):
        """Test processing large number of items."""
        inputs = [f"item_{i}" for i in range(100)]
        result = transform.process(inputs)
        
        assert len(result.data) == 100
        assert all(r == i.upper() for i, r in zip(inputs, result.data))
        # With batch size 3, we should have 34 batches (33 full + 1 partial)
        assert transform.provider.call_count == 34
    
    def test_custom_process_function(self, mock_store, config):
        """Test with custom processing function."""
        # Provider that reverses strings
        provider = MockProvider(process_func=lambda items: [item[::-1] for item in items])
        transform = BatchInferenceTransform(
            provider=provider,
            store=mock_store,
            config=config
        )
        
        inputs = ["hello", "world"]
        result = transform.process(inputs)
        
        assert result.data == ["olleh", "dlrow"]
        assert result.success
    
    def test_partial_batch_retry(self, mock_store):
        """Test retry with different batch sizes."""
        config = BatchConfig(
            batch_size=2,
            hash_function=simple_hash,
            max_retries=3,
            retry_delay=0.01
        )
        
        # Fail on first batch only
        call_count = 0
        def failing_process(items):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and len(items) == 2:
                raise Exception("First batch failed")
            return [item.upper() for item in items]
        
        provider = Mock()
        provider.process_batch = failing_process
        
        transform = BatchInferenceTransform(
            provider=provider,
            store=mock_store,
            config=config
        )
        
        inputs = ["a", "b", "c"]
        result = transform.process(inputs)
        
        assert result.data == ["A", "B", "C"]
        assert call_count == 3  # Failed once on first batch, then succeeded
    
    def test_input_output_types(self, transform):
        """Test component type declarations."""
        assert transform.input_types == [list]
        assert transform.output_types == [list]
    
    def test_component_name(self, mock_provider, mock_store, config):
        """Test component naming."""
        # Default name
        transform1 = BatchInferenceTransform(
            provider=mock_provider,
            store=mock_store,
            config=config
        )
        assert transform1.name == "BatchInferenceTransform"
        
        # Custom name
        transform2 = BatchInferenceTransform(
            provider=mock_provider,
            store=mock_store,
            config=config,
            name="CustomBatch"
        )
        assert transform2.name == "CustomBatch"
    
    def test_execution_time_tracking(self, transform):
        """Test that execution time is tracked."""
        inputs = ["a", "b", "c"]
        result = transform.process(inputs)
        
        assert result.execution_time_ms > 0
        assert result.timestamp is not None