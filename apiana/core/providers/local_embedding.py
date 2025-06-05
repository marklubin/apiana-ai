"""
Local embedding provider using sentence-transformers library.
"""

import logging
from typing import List, Dict, Any

try:
    import torch
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from apiana.core.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class LocalTransformersEmbedding(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers library.
    
    Provides embeddings using local transformer models optimized for
    semantic similarity tasks.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        **kwargs
    ):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers library is required for LocalTransformersEmbedding")
        
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        logger.info(f"Loading embedding model {model_name} on {device}...")
        
        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        
        # Get embedding dimension
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding model {model_name} loaded successfully (dim={self._embedding_dim})")
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text query."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_tensor=False
        )
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            batch_size=self.batch_size,
            convert_to_tensor=False,
            show_progress_bar=len(texts) > 100
        )
        
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self._embedding_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self._embedding_dim,
            "normalize_embeddings": self.normalize_embeddings,
            "max_seq_length": self.model.max_seq_length,
            "batch_size": self.batch_size
        }