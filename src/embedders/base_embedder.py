from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List


class BaseEmbedder(ABC):
    """Abstract base class for all embedders."""

    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Initialize the embedder.

        Args:
            model_name: Name/path of the model to use
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.device = device
        self.model = None

    @abstractmethod
    def load_model(self):
        """Load the embedding model."""
        pass

    @abstractmethod
    def embed(self, content: Union[str, object]) -> np.ndarray:
        """
        Generate embedding vector for single input.

        Args:
            content: Input content (text, image path, etc.)

        Returns:
            Embedding vector as numpy array
        """
        pass

    @abstractmethod
    def batch_embed(self, contents: List) -> np.ndarray:
        """
        Generate embeddings for batch of inputs.

        Args:
            contents: List of inputs

        Returns:
            Matrix of embedding vectors
        """
        pass

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.embedding_dim
