"""Unit tests for LocalTransformersEmbedding provider."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

# Mock sentence_transformers before import
mock_sentence_transformer = MagicMock()
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False

with patch.dict('sys.modules', {
    'sentence_transformers': mock_sentence_transformer,
    'torch': mock_torch
}):
    from apiana.core.providers.local_embedding import LocalTransformersEmbedding


class TestLocalTransformersEmbedding:
    """Tests for LocalTransformersEmbedding provider."""
    
    @patch('apiana.core.providers.local_embedding.HAS_SENTENCE_TRANSFORMERS', True)
    @patch('apiana.core.providers.local_embedding.SentenceTransformer')
    @patch('apiana.core.providers.local_embedding.torch')
    def test_initialization(self, mock_torch, mock_st_class):
        """Test provider initialization."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model
        
        # Create provider
        provider = LocalTransformersEmbedding(
            model_name="test-model",
            device="cpu",
            normalize_embeddings=True,
            batch_size=16
        )
        
        # Verify initialization
        assert provider.model_name == "test-model"
        assert provider.device == "cpu"
        assert provider.normalize_embeddings == True
        assert provider.batch_size == 16
        assert provider._embedding_dim == 384
        
        mock_st_class.assert_called_once_with("test-model", device="cpu")
    
    @patch('apiana.core.providers.local_embedding.HAS_SENTENCE_TRANSFORMERS', True)
    @patch('apiana.core.providers.local_embedding.SentenceTransformer')
    @patch('apiana.core.providers.local_embedding.torch')
    def test_auto_device_selection(self, mock_torch, mock_st_class):
        """Test automatic device selection."""
        # Test with CUDA available
        mock_torch.cuda.is_available.return_value = True
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model
        
        provider = LocalTransformersEmbedding(device="auto")
        assert provider.device == "cuda"
        
        # Test without CUDA
        mock_torch.cuda.is_available.return_value = False
        provider = LocalTransformersEmbedding(device="auto")
        assert provider.device == "cpu"
    
    @patch('apiana.core.providers.local_embedding.HAS_SENTENCE_TRANSFORMERS', False)
    def test_missing_dependencies(self):
        """Test error when dependencies are missing."""
        with pytest.raises(ImportError, match="sentence-transformers library is required"):
            LocalTransformersEmbedding()
    
    @patch('apiana.core.providers.local_embedding.HAS_SENTENCE_TRANSFORMERS', True)
    @patch('apiana.core.providers.local_embedding.SentenceTransformer')
    @patch('apiana.core.providers.local_embedding.torch')
    def test_embed_query(self, mock_torch, mock_st_class):
        """Test embedding a single query."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = mock_embedding
        mock_st_class.return_value = mock_model
        
        # Create provider and embed query
        provider = LocalTransformersEmbedding()
        result = provider.embed_query("Test query")
        
        # Verify
        assert result == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with(
            "Test query",
            normalize_embeddings=True,
            convert_to_tensor=False
        )
    
    @patch('apiana.core.providers.local_embedding.HAS_SENTENCE_TRANSFORMERS', True)
    @patch('apiana.core.providers.local_embedding.SentenceTransformer')
    @patch('apiana.core.providers.local_embedding.torch')
    def test_embed_documents(self, mock_torch, mock_st_class):
        """Test embedding multiple documents."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_st_class.return_value = mock_model
        
        # Create provider and embed documents
        provider = LocalTransformersEmbedding(batch_size=2)
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        result = provider.embed_documents(texts)
        
        # Verify
        assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_model.encode.assert_called_once_with(
            texts,
            normalize_embeddings=True,
            batch_size=2,
            convert_to_tensor=False,
            show_progress_bar=False
        )
    
    @patch('apiana.core.providers.local_embedding.HAS_SENTENCE_TRANSFORMERS', True)
    @patch('apiana.core.providers.local_embedding.SentenceTransformer')
    @patch('apiana.core.providers.local_embedding.torch')
    def test_embed_documents_empty(self, mock_torch, mock_st_class):
        """Test embedding empty document list."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model
        
        # Create provider and embed empty list
        provider = LocalTransformersEmbedding()
        result = provider.embed_documents([])
        
        # Verify
        assert result == []
        mock_model.encode.assert_not_called()
    
    @patch('apiana.core.providers.local_embedding.HAS_SENTENCE_TRANSFORMERS', True)
    @patch('apiana.core.providers.local_embedding.SentenceTransformer')
    @patch('apiana.core.providers.local_embedding.torch')
    def test_embed_documents_progress_bar(self, mock_torch, mock_st_class):
        """Test progress bar shown for large batches."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.zeros((101, 384))
        mock_model.encode.return_value = mock_embeddings
        mock_st_class.return_value = mock_model
        
        # Create provider and embed many documents
        provider = LocalTransformersEmbedding()
        texts = ["Doc"] * 101
        result = provider.embed_documents(texts)
        
        # Verify progress bar was enabled
        mock_model.encode.assert_called_once()
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs['show_progress_bar'] == True
    
    @patch('apiana.core.providers.local_embedding.HAS_SENTENCE_TRANSFORMERS', True)
    @patch('apiana.core.providers.local_embedding.SentenceTransformer')
    @patch('apiana.core.providers.local_embedding.torch')
    def test_get_embedding_dimension(self, mock_torch, mock_st_class):
        """Test getting embedding dimension."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st_class.return_value = mock_model
        
        # Create provider
        provider = LocalTransformersEmbedding()
        
        # Verify
        assert provider.get_embedding_dimension() == 768
    
    @patch('apiana.core.providers.local_embedding.HAS_SENTENCE_TRANSFORMERS', True)
    @patch('apiana.core.providers.local_embedding.SentenceTransformer')
    @patch('apiana.core.providers.local_embedding.torch')
    def test_get_model_info(self, mock_torch, mock_st_class):
        """Test getting model information."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.max_seq_length = 512
        mock_st_class.return_value = mock_model
        
        # Create provider
        provider = LocalTransformersEmbedding(
            model_name="test-model",
            device="cpu",
            normalize_embeddings=False,
            batch_size=64
        )
        
        # Get info
        info = provider.get_model_info()
        
        # Verify
        assert info == {
            "model_name": "test-model",
            "device": "cpu",
            "embedding_dimension": 384,
            "normalize_embeddings": False,
            "max_seq_length": 512,
            "batch_size": 64
        }