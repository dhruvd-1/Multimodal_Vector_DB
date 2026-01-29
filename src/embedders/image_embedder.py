"""
Image Embedder using CLIP vision encoder - PRODUCTION IMPLEMENTATION

Selected Model: CLIP ViT-B/32 with LAION-2B weights
- Speed: 6.3ms per image
- Accuracy: 95.2% R@10
- Compression: FP16 with 0% accuracy drop
- Mobile ready: 150MB vision encoder

Based on Week 1 evaluation results.
"""

import sys
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .base_embedder import BaseEmbedder, QuantizationType, ModelFormat
from .projection import ProjectionLayer


class ImageEmbedder(BaseEmbedder):
    """Image embedder using CLIP ViT-B/32 (LAION-2B) vision encoder."""

    # Week 1 winner: CLIP ViT-B/32 with LAION-2B weights
    DEFAULT_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    FALLBACK_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        model_name: str = None,
        device: str = 'cpu',
        quantization: str = 'fp16',  # Default FP16 per specs
        model_format: str = 'pytorch',
        max_batch_size: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
        enable_async: bool = False,
        use_projection: bool = False,
        projection_dim: int = 1024,
    ):
        """
        Initialize image embedder with Week 1 winning configuration.

        Args:
            model_name: CLIP model (default: LAION ViT-B/32 - 6.3ms, 95.2% R@10)
            device: Device ('cpu', 'cuda')
            quantization: Quantization ('fp16' recommended, 'none', 'int8')
            model_format: Model format ('pytorch')
            max_batch_size: Max batch size for mobile
            memory_limit_mb: Memory limit
            enable_async: Enable async operations
            use_projection: Use 512D->1024D projection (Month 2 feature)
            projection_dim: Projection output dimension
        """
        model_name = model_name or self.DEFAULT_MODEL

        super().__init__(
            model_name=model_name,
            device=device,
            quantization=quantization,
            model_format=model_format,
            max_batch_size=max_batch_size,
            memory_limit_mb=memory_limit_mb,
            enable_async=enable_async,
        )

        self.processor = None
        self.use_projection = use_projection
        self.projection_dim = projection_dim
        self.projection = None

        # Set embedding dimension
        self.embedding_dim = projection_dim if use_projection else 512

    def load_model(self) -> None:
        """Load CLIP model with Week 1 validated configuration."""
        print(f"Loading CLIP vision encoder: {self.model_name}")

        try:
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Failed to load {self.model_name}, using fallback: {e}")
            self.model_name = self.FALLBACK_MODEL
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

        # Apply quantization (FP16 validated: 0% accuracy drop)
        if self.quantization == QuantizationType.FP16:
            if self.device == 'cuda':
                self.model = self.model.half()
                print("Applied FP16 quantization (validated: 0% accuracy drop)")
        elif self.quantization == QuantizationType.INT8:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                print("Applied INT8 quantization")
            except Exception as e:
                print(f"INT8 quantization failed: {e}")

        self.model.to(self.device)
        self.model.eval()

        # Initialize projection (Month 2 feature)
        if self.use_projection:
            self.projection = ProjectionLayer(input_dim=512, output_dim=self.projection_dim)
            self.projection.to(self.device)
            self.projection.eval()
            print(f"Initialized projection: 512D -> {self.projection_dim}D")

        self._is_loaded = True

        status = f"✓ Model loaded on {self.device}"
        if self.quantization != QuantizationType.NONE:
            status += f" ({self.quantization.value})"
        if self.use_projection:
            status += f" + projection"
        if self._is_mobile:
            status += " [MOBILE MODE]"
        print(status)

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.projection is not None:
            del self.projection
            self.projection = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        super().unload_model()

    def embed(self, image_input: Union[str, Image.Image]) -> np.ndarray:
        """
        Generate embedding for single image.

        Args:
            image_input: Path to image or PIL Image

        Returns:
            Normalized embedding vector [512D or projection_dim]
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        print(">>> RUNNING UPDATED EMBED() <<<")
        # Load image if path provided
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            # Get image features through visual projection (512D for ViT-B/32)
            vision_outputs = self.model.vision_model(**inputs)
            pooled_output = vision_outputs.pooler_output  # [batch, 768]
            image_embeds = self.model.visual_projection(pooled_output)  # [batch, 512]

            # Normalize
            embeddings = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            # Apply projection if configured
            if self.use_projection and self.projection is not None:
                embeddings = self.projection(embeddings)

        return embeddings.cpu().numpy()[0]

    def batch_embed(
        self,
        images: List[Union[str, Image.Image]],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings for batch of images.

        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size (respects max_batch_size)

        Returns:
            Matrix of normalized embeddings [N, 512] or [N, projection_dim]
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        batch_size = self.get_effective_batch_size(batch_size)
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Load images
            pil_images = []
            for img in batch:
                if isinstance(img, str):
                    pil_images.append(Image.open(img).convert("RGB"))
                else:
                    pil_images.append(img.convert("RGB"))

            # Process batch
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                # Get image features through visual projection (512D for ViT-B/32)
                vision_outputs = self.model.vision_model(**inputs)
                pooled_output = vision_outputs.pooler_output  # [batch, 768]
                image_embeds = self.model.visual_projection(pooled_output)  # [batch, 512]

                # Normalize
                embeddings = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

                # Apply projection if configured
                if self.use_projection and self.projection is not None:
                    embeddings = self.projection(embeddings)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def warmup(self, sample_input: Optional[Union[str, Image.Image]] = None) -> None:
        """
        Warm up the model with a sample inference.

        Args:
            sample_input: Optional sample image (creates dummy if not provided)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        sample = sample_input or Image.new('RGB', (224, 224))
        _ = self.embed(sample)
        print("Image embedder warmup complete")

    def load_projection_weights(self, path: str):
        """Load pre-trained projection weights."""
        if not self.use_projection:
            raise RuntimeError("Projection mode not enabled")

        self.projection = ProjectionLayer.load(path, self.device)
        print(f"Loaded projection weights from {path}")

    def save_projection_weights(self, path: str):
        """Save projection weights."""
        if not self.use_projection or self.projection is None:
            raise RuntimeError("No projection to save")

        self.projection.save(path)


# =============================================================================
# Standalone Test
# =============================================================================

def test_image_embedder():
    """Test the image embedder with Week 1 configuration."""
    print("=" * 60)
    print("IMAGE EMBEDDER TEST (Week 1 Winner: CLIP ViT-B/32 LAION-2B)")
    print("=" * 60)

    # Test 1: Basic embedder
    print("\n--- Test 1: Load Model ---")
    embedder = ImageEmbedder(device="cpu")
    embedder.load_model()

    # Test 2: Single image embedding
    print("\n--- Test 2: Single Image Embedding ---")
    test_img = Image.new('RGB', (224, 224), color='red')
    embedding = embedder.embed(test_img)
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding norm: {np.linalg.norm(embedding):.6f}")
    assert embedding.shape == (512,)
    assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-5

    # Test 3: Batch embedding
    print("\n--- Test 3: Batch Image Embedding ---")
    images = [
        Image.new('RGB', (224, 224), color='red'),
        Image.new('RGB', (224, 224), color='green'),
        Image.new('RGB', (224, 224), color='blue'),
    ]
    embeddings = embedder.batch_embed(images)
    print(f"✓ Batch shape: {embeddings.shape}")
    assert embeddings.shape == (3, 512)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)

    # Test 4: Similarity
    print("\n--- Test 4: Image Similarity ---")
    red1 = Image.new('RGB', (224, 224), color=(255, 0, 0))
    red2 = Image.new('RGB', (224, 224), color=(200, 50, 50))
    blue = Image.new('RGB', (224, 224), color=(0, 0, 255))

    emb_red1 = embedder.embed(red1)
    emb_red2 = embedder.embed(red2)
    emb_blue = embedder.embed(blue)

    sim_red_red = np.dot(emb_red1, emb_red2)
    sim_red_blue = np.dot(emb_red1, emb_blue)

    print(f"Similarity (red vs similar red): {sim_red_red:.4f}")
    print(f"Similarity (red vs blue): {sim_red_blue:.4f}")
    assert sim_red_red > sim_red_blue
    print("✓ Similar colors have higher similarity")

    embedder.unload_model()

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_image_embedder()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
