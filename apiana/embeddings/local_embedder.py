"""
Local embedding generation using HuggingFace Transformers.

Runs embedding models in-process for privacy and control.
"""

import asyncio
import logging
from typing import List, Union, Optional
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Local embedding generator using sentence-transformers.
    
    Supports various embedding models and runs them in-process.
    """
    
    # Recommended models with their dimensions
    MODEL_INFO = {
        "all-MiniLM-L6-v2": {
            "dim": 384,
            "max_tokens": 256,
            "description": "Fast, good for general use"
        },
        "all-mpnet-base-v2": {
            "dim": 768,
            "max_tokens": 384,
            "description": "Best quality for sentence embeddings"
        },
        "bge-base-en-v1.5": {
            "dim": 768,
            "max_tokens": 512,
            "description": "BAAI model, excellent for retrieval"
        },
        "bge-small-en-v1.5": {
            "dim": 384,
            "max_tokens": 512,
            "description": "Smaller BAAI model, good balance"
        },
        "nomic-embed-text-v1.5": {
            "dim": 768,
            "max_tokens": 8192,
            "description": "Long context, variable dimensions"
        }
    }
    
    def __init__(
        self,
        model_name: str = "bge-base-en-v1.5",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        normalize: bool = True
    ):
        """
        Initialize local embedder.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on (None for auto-detect)
            cache_dir: Directory to cache models
            normalize: Whether to normalize embeddings to unit length
        """
        self.model_name = model_name
        self.normalize = normalize
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
        self.device = device
        
        # Load model
        logger.info(f"Loading embedding model '{model_name}' on {device}")
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(cache_dir) if cache_dir else None
        )
        
        # Get model info
        self.embedding_dim = self.MODEL_INFO.get(model_name, {}).get("dim", 768)
        self.max_tokens = self.MODEL_INFO.get(model_name, {}).get("max_tokens", 512)
        
        logger.info(f"Model loaded: dim={self.embedding_dim}, max_tokens={self.max_tokens}")
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s).
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        
        # Encode
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        
        # Convert to list format for storage
        if single_input:
            return embeddings[0].tolist()
        else:
            return [emb.tolist() for emb in embeddings]
    
    async def embed_async(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings asynchronously using default executor.
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, text)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Embed multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return [emb.tolist() for emb in embeddings]
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Convert to numpy if needed
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Normalize if not already normalized
        if not self.normalize:
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)
        
        return float(np.dot(emb1, emb2))
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_tokens": self.max_tokens,
            "device": self.device,
            "normalize": self.normalize,
            "description": self.MODEL_INFO.get(self.model_name, {}).get("description", "Unknown model")
        }
    
    @classmethod
    def list_available_models(cls) -> dict:
        """
        List recommended models with their properties.
        
        Returns:
            Dictionary of model information
        """
        return cls.MODEL_INFO


# Factory functions for common use cases

def create_fast_embedder(device: Optional[str] = None) -> LocalEmbedder:
    """Create a fast embedder for real-time use."""
    return LocalEmbedder(
        model_name="all-MiniLM-L6-v2",
        device=device
    )


def create_quality_embedder(device: Optional[str] = None) -> LocalEmbedder:
    """Create a high-quality embedder for accuracy."""
    return LocalEmbedder(
        model_name="bge-base-en-v1.5",
        device=device
    )


def create_long_context_embedder(device: Optional[str] = None) -> LocalEmbedder:
    """Create an embedder for long texts."""
    return LocalEmbedder(
        model_name="nomic-embed-text-v1.5",
        device=device
    )