"""
Local embedding generation using HuggingFace Transformers.

Runs embedding models in-process for privacy and control.
"""

import logging
from typing import List, Union
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


DIM = 768
MAX_TOKENS = 512


class Embedder:
    """
    Local embedding generator using sentence-transformers.

    Supports various embedding models and runs them in-process.
    """

    def __init__(
        self, model_name: str, device: str, cache_dir: Path, normalize: bool = True
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
        self.device = device

        # Load model
        logger.info(f"Loading embedding model '{model_name}' on {device}")
        self.model = SentenceTransformer(
            model_name, device=device, cache_folder=str(cache_dir)
        )

        logger.info("Model loaded!")

    def embed(self, texts: List[str]):
        # Encode
        # embeddings = self.model.encode(
        #     texts,
        #     normalize_embeddings=self.normalize,
        #     show_progress_bar=len(texts) > 100,
        #     convert_to_numpy=True,
        # )
        pass

    async def embed_async(
        self, text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings asynchronously using default executor.

        Args:
            text: Single text or list of texts

        Returns:
            Single embedding vector or list of embedding vectors
        #"""
        # loop = asyncio.get_event_loop()
        # return await loop.run_in_executor(None, self.embed, text)
        #
        return []

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
        # return {
        #     "model_name": self.model_name,
        #     "embedding_dimension": self.embedding_dim,
        #     "max_tokens": self.max_tokens,
        #     "device": self.device,
        #     "normalize": self.normalize,
        #     "description": self.MODEL_INFO.get(self.model_name, {}).get(
        #         "description", "Unknown model"
        #     ),
        # }

        return {}
