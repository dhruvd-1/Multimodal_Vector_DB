from transformers import CLIPModel, CLIPTokenizer
import torch
import numpy as np
from typing import List, Union
from .base_embedder import BaseEmbedder


class TextEmbedder(BaseEmbedder):
    """Text embedder using CLIP text encoder."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = 'cpu', max_length: int = 77):
        """
        Initialize text embedder.

        Args:
            model_name: CLIP model name
            device: Device to run on
            max_length: Maximum token length
        """
        super().__init__(model_name, device)
        self.max_length = max_length
        self.tokenizer = None
        self.embedding_dim = 512  # CLIP base dimension

    def load_model(self):
        """Load CLIP model and tokenizer."""
        print(f"Loading CLIP model: {self.model_name}")
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Input text string

        Returns:
            Embedding vector (normalized)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
            # Normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy()[0]

    def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for batch of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            Matrix of embedding vectors
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self.model.get_text_features(**inputs)
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
