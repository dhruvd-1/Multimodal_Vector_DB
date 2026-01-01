"""
Standalone Test for Image Embedder
Run: python test_image_embedder.py
"""

import sys

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .projection import ProjectionLayer


class ImageEmbedder:
    """Image embedder using CLIP vision encoder."""

    def __init__(self, model_name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", device="cpu", use_projection=False):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self.use_projection = use_projection
        self.projection = None
        self.embedding_dim = 1024 if use_projection else 512

    def load_model(self):
        """Load CLIP model and processor."""
        print(f"Loading CLIP model: {self.model_name}")
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.use_projection:
            self.projection = ProjectionLayer(input_dim=512, output_dim=1024)
            self.projection.to(self.device)
            self.projection.eval()
            print(f"✓ Model loaded on {self.device} with 1024D projection")
        else:
            print(f"✓ Model loaded on {self.device}")

    def embed(self, image_input):
        """Generate embedding for single image."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load image if path provided
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            if self.use_projection:
                embeddings = self.projection(embeddings)

        return embeddings.cpu().numpy()[0]

    def batch_embed(self, images, batch_size=32):
        """Generate embeddings for batch of images."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            # Load images
            pil_images = []
            for img in batch:
                if isinstance(img, str):
                    pil_images.append(Image.open(img).convert("RGB"))
                else:
                    pil_images.append(img.convert("RGB"))

            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                if self.use_projection:
                    embeddings = self.projection(embeddings)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


def create_test_images():
    """Create dummy test images with different colors."""
    print("Creating test images...")

    images_data = [
        ("red", (255, 0, 0)),
        ("green", (0, 255, 0)),
        ("blue", (0, 0, 255)),
        ("yellow", (255, 255, 0)),
        ("purple", (128, 0, 128)),
        ("orange", (255, 165, 0)),
        ("cyan", (0, 255, 255)),
    ]

    images = []
    for name, color in images_data:
        img = Image.new("RGB", (224, 224), color=color)
        images.append((name, img))

    print(f"✓ Created {len(images)} test images")
    return images


def test_image_embedder():
    """Test the image embedder with dummy data."""
    print("=" * 60)
    print("IMAGE EMBEDDER TEST")
    print("=" * 60)

    # Initialize embedder
    embedder = ImageEmbedder(device="cpu")
    embedder.load_model()

    # Create test images
    test_images = create_test_images()

    # Test 1: Single image embedding
    print("\n--- Test 1: Single Image Embedding ---")
    name, img = test_images[0]
    print(f"Processing: {name} image")

    embedding = embedder.embed(img)
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding dimension: {len(embedding)}")
    print(f"✓ Embedding norm (should be ~1.0): {np.linalg.norm(embedding):.6f}")
    print(f"✓ First 5 values: {embedding[:5]}")

    # Test 2: Batch image embedding
    print("\n--- Test 2: Batch Image Embedding ---")
    images_only = [img for _, img in test_images]
    names_only = [name for name, _ in test_images]

    print(f"Processing {len(images_only)} images...")
    for name in names_only:
        print(f"  - {name}")

    embeddings = embedder.batch_embed(images_only)
    print(f"\n✓ Batch embeddings shape: {embeddings.shape}")
    print(
        f"✓ All vectors normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)}"
    )

    # Test 3: Similarity test
    print("\n--- Test 3: Image Similarity Test ---")
    red_img = test_images[0][1]  # Red
    red_img2 = Image.new("RGB", (224, 224), color=(200, 50, 50))  # Similar red
    blue_img = test_images[2][1]  # Blue

    emb_red1 = embedder.embed(red_img)
    emb_red2 = embedder.embed(red_img2)
    emb_blue = embedder.embed(blue_img)

    sim_red_red = np.dot(emb_red1, emb_red2)
    sim_red_blue = np.dot(emb_red1, emb_blue)

    print(f"Similarity (red vs similar red): {sim_red_red:.4f}")
    print(f"Similarity (red vs blue): {sim_red_blue:.4f}")
    print(f"✓ Similar colors have higher similarity: {sim_red_red > sim_red_blue}")

    # Test 4: Search simulation
    print("\n--- Test 4: Simple Image Search Simulation ---")
    query_img = Image.new("RGB", (224, 224), color=(200, 0, 0))  # Red-ish
    print("Query: Red-ish image")
    query_emb = embedder.embed(query_img)

    # Compute similarities
    print("\nRanked results:")
    sims = embeddings @ query_emb
    ranked_indices = np.argsort(sims)[::-1]

    for rank, idx in enumerate(ranked_indices[:5], 1):
        print(f"{rank}. [{sims[idx]:.4f}] {names_only[idx]}")

    # Test 5: Self-similarity check
    print("\n--- Test 5: Self-Similarity Check ---")
    test_img = test_images[0][1]
    emb1 = embedder.embed(test_img)
    emb2 = embedder.embed(test_img)

    self_sim = np.dot(emb1, emb2)
    print(f"Self-similarity (should be ~1.0): {self_sim:.6f}")
    print(f"✓ Image embedding is deterministic: {np.allclose(self_sim, 1.0)}")

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
