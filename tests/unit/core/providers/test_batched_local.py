"""Unit tests for batched local providers."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import torch

from apiana.core.providers.base import LLMResponse


class TestBatchedLocalTransformersLLM:
    """Tests for BatchedLocalTransformersLLM."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LocalTransformersLLM."""
        provider = Mock()
        provider.model_name = "test-model"
        provider.max_length = 2048
        
        # Mock model
        model = Mock()
        model.parameters.return_value = iter([Mock(device=torch.device('cpu'))])
        provider.model = model
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        provider.tokenizer = tokenizer
        
        # Mock generation config
        gen_config = Mock()
        provider.generation_config = gen_config
        
        return provider
    
    def test_initialization(self, mock_llm_provider):
        """Test initialization with LLM provider."""
        from apiana.core.providers.batched_local import BatchedLocalTransformersLLM
        
        batched = BatchedLocalTransformersLLM(mock_llm_provider)
        
        assert batched.llm_provider == mock_llm_provider
        assert batched.model == mock_llm_provider.model
        assert batched.tokenizer == mock_llm_provider.tokenizer
        assert batched.generation_config == mock_llm_provider.generation_config
        assert batched.max_length == 2048
    
    def test_process_batch_empty(self, mock_llm_provider):
        """Test processing empty batch."""
        from apiana.core.providers.batched_local import BatchedLocalTransformersLLM
        
        batched = BatchedLocalTransformersLLM(mock_llm_provider)
        result = batched.process_batch([])
        
        assert result == []
    
    def test_process_batch_single_item(self, mock_llm_provider):
        """Test processing single item uses regular invoke."""
        from apiana.core.providers.batched_local import BatchedLocalTransformersLLM
        
        # Mock invoke response
        mock_response = LLMResponse(
            content="Test response",
            model="test-model",
            usage={"total_tokens": 10}
        )
        mock_llm_provider.invoke.return_value = mock_response
        
        batched = BatchedLocalTransformersLLM(mock_llm_provider)
        result = batched.process_batch(["Test prompt"])
        
        assert len(result) == 1
        assert result[0] == mock_response
        mock_llm_provider.invoke.assert_called_once_with("Test prompt")
    
    @patch('apiana.core.providers.batched_local.torch')
    def test_process_batch_multiple_items(self, mock_torch, mock_llm_provider):
        """Test processing multiple items in batch."""
        from apiana.core.providers.batched_local import BatchedLocalTransformersLLM
        
        # Mock tokenizer
        input_ids = torch.tensor([[1, 2, 3], [1, 2, 0], [1, 0, 0]])  # Different lengths
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
        mock_llm_provider.tokenizer.return_value = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Mock model generate
        outputs = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 0, 6, 7], [1, 0, 0, 8, 9]])
        mock_llm_provider.model.generate.return_value = outputs
        
        # Mock decode
        mock_llm_provider.tokenizer.decode.side_effect = ["Response 1", "Response 2", "Response 3"]
        
        # Create batched provider
        batched = BatchedLocalTransformersLLM(mock_llm_provider)
        
        # Process batch
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        result = batched.process_batch(prompts)
        
        # Verify
        assert len(result) == 3
        assert all(isinstance(r, LLMResponse) for r in result)
        assert result[0].content == "Response 1"
        assert result[1].content == "Response 2"
        assert result[2].content == "Response 3"
        
        # Check metadata
        for i, response in enumerate(result):
            assert response.model == "test-model"
            assert response.metadata["batch_index"] == i
            assert response.metadata["batch_size"] == 3
    
    def test_process_batch_with_padding(self, mock_llm_provider):
        """Test that tokenizer is called with proper padding."""
        from apiana.core.providers.batched_local import BatchedLocalTransformersLLM
        
        # Setup minimal mocks for batch processing
        mock_llm_provider.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2], [1, 3]]),
            'attention_mask': torch.tensor([[1, 1], [1, 1]])
        }
        mock_llm_provider.model.generate.return_value = torch.tensor([[1, 2, 4], [1, 3, 5]])
        mock_llm_provider.tokenizer.decode.side_effect = ["Out1", "Out2"]
        
        batched = BatchedLocalTransformersLLM(mock_llm_provider)
        batched.process_batch(["P1", "P2"])
        
        # Verify tokenizer called with correct arguments
        mock_llm_provider.tokenizer.assert_called_once_with(
            ["P1", "P2"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1548  # 2048 - 500
        )


class TestBatchedLocalTransformersEmbedding:
    """Tests for BatchedLocalTransformersEmbedding."""
    
    @pytest.fixture
    def mock_embedding_provider(self):
        """Create a mock LocalTransformersEmbedding."""
        provider = Mock()
        provider.embed_documents = Mock()
        return provider
    
    def test_initialization(self, mock_embedding_provider):
        """Test initialization with embedding provider."""
        from apiana.core.providers.batched_local import BatchedLocalTransformersEmbedding
        
        batched = BatchedLocalTransformersEmbedding(mock_embedding_provider)
        assert batched.embedding_provider == mock_embedding_provider
    
    def test_process_batch_empty(self, mock_embedding_provider):
        """Test processing empty batch."""
        from apiana.core.providers.batched_local import BatchedLocalTransformersEmbedding
        
        batched = BatchedLocalTransformersEmbedding(mock_embedding_provider)
        result = batched.process_batch([])
        
        assert result == []
        mock_embedding_provider.embed_documents.assert_not_called()
    
    def test_process_batch_delegates_to_provider(self, mock_embedding_provider):
        """Test that batch processing delegates to provider."""
        from apiana.core.providers.batched_local import BatchedLocalTransformersEmbedding
        
        # Mock embeddings
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_embedding_provider.embed_documents.return_value = embeddings
        
        batched = BatchedLocalTransformersEmbedding(mock_embedding_provider)
        texts = ["Text 1", "Text 2", "Text 3"]
        result = batched.process_batch(texts)
        
        assert result == embeddings
        mock_embedding_provider.embed_documents.assert_called_once_with(texts)


class TestBatchedLLMWithSystemPrompt:
    """Tests for BatchedLLMWithSystemPrompt."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LocalTransformersLLM."""
        provider = Mock()
        provider.model_name = "test-model"
        provider.max_length = 2048
        
        # Mock model
        model = Mock()
        model.parameters.return_value = iter([Mock(device=torch.device('cpu'))])
        provider.model = model
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        provider.tokenizer = tokenizer
        
        # Mock generation config
        gen_config = Mock()
        provider.generation_config = gen_config
        
        return provider
    
    def test_initialization(self, mock_llm_provider):
        """Test initialization."""
        from apiana.core.providers.batched_local import BatchedLLMWithSystemPrompt
        
        batched = BatchedLLMWithSystemPrompt(mock_llm_provider)
        assert batched.llm_provider == mock_llm_provider
        assert batched.batched_provider is not None
    
    def test_process_batch_empty(self, mock_llm_provider):
        """Test processing empty batch."""
        from apiana.core.providers.batched_local import BatchedLLMWithSystemPrompt
        
        batched = BatchedLLMWithSystemPrompt(mock_llm_provider)
        result = batched.process_batch([])
        
        assert result == []
    
    @patch('apiana.core.providers.batched_local.BatchedLocalTransformersLLM')
    def test_process_batch_with_system_prompts(self, mock_batched_class, mock_llm_provider):
        """Test processing with system prompts."""
        from apiana.core.providers.batched_local import BatchedLLMWithSystemPrompt
        
        # Mock the batched provider
        mock_batched_instance = Mock()
        mock_responses = [
            LLMResponse(content="R1", model="test"),
            LLMResponse(content="R2", model="test"),
            LLMResponse(content="R3", model="test")
        ]
        mock_batched_instance.process_batch.return_value = mock_responses
        mock_batched_class.return_value = mock_batched_instance
        
        # Create provider
        batched = BatchedLLMWithSystemPrompt(mock_llm_provider)
        
        # Process items with mixed system prompts
        items = [
            {"prompt": "Hello", "system_instruction": "Be friendly"},
            {"prompt": "Calculate 2+2"},
            {"prompt": "Goodbye", "system_instruction": "Be formal"}
        ]
        
        result = batched.process_batch(items)
        
        # Verify formatted prompts
        expected_prompts = [
            "System: Be friendly\n\nUser: Hello\n\nAssistant:",
            "Calculate 2+2",
            "System: Be formal\n\nUser: Goodbye\n\nAssistant:"
        ]
        
        mock_batched_instance.process_batch.assert_called_once_with(expected_prompts)
        assert result == mock_responses
    
    def test_process_batch_handles_missing_keys(self, mock_llm_provider):
        """Test handling of missing keys in input dicts."""
        from apiana.core.providers.batched_local import BatchedLLMWithSystemPrompt
        
        with patch('apiana.core.providers.batched_local.BatchedLocalTransformersLLM') as mock_batched:
            mock_instance = Mock()
            mock_instance.process_batch.return_value = [LLMResponse(content="", model="test")]
            mock_batched.return_value = mock_instance
            
            batched = BatchedLLMWithSystemPrompt(mock_llm_provider)
            
            # Test with missing keys
            items = [{}]  # No prompt or system_instruction
            result = batched.process_batch(items)
            
            # Should use empty string for missing prompt
            mock_instance.process_batch.assert_called_once_with([''])