from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from typing import List, Union
from .base_embedder import BaseEmbedder


class ImageEmbedder(BaseEmbedder):
    """Image embedder using CLIP vision encoder."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = 'cpu'):
        """
        Initialize image embedder.

        Args:
            model_name: CLIP model name
            device: Device to run on
        """
        super().__init__(model_name, device)
        self.processor = None
        self.embedding_dim = 512  # CLIP base dimension

    def load_model(self):
        """Load CLIP model and processor."""
        print(f"Loading CLIP model: {self.model_name}")
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def embed(self, image_input: Union[str, Image.Image]) -> np.ndarray:
        """
        Generate embedding for single image.

        Args:
            image_input: Image file path or PIL Image

        Returns:
            Embedding vector (normalized)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load image if path provided
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
            # Normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy()[0]

    def batch_embed(self, images: List[Union[str, Image.Image]],
                   batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for batch of images.

        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for processing

        Returns:
            Matrix of embedding vectors
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Load images
            pil_images = []
            for img in batch:
                if isinstance(img, str):
                    pil_images.append(Image.open(img).convert('RGB'))
                else:
                    pil_images.append(img.convert('RGB'))

            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
