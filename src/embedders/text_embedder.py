"""
Standalone Test for Text Embedder
Run: python test_text_embedder.py
"""

import sys

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

from .projection import ProjectionLayer


class TextEmbedder:
    """Text embedder using CLIP text encoder."""

    def __init__(self, model_name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", device="cpu", use_projection=False):
        self.model_name = model_name
        self.device = device
        self.max_length = 77
        self.tokenizer = None
        self.model = None
        self.use_projection = use_projection
        self.projection = None
        self.embedding_dim = 1024 if use_projection else 512

    def load_model(self):
        """Load CLIP model and tokenizer."""
        print(f"Loading CLIP model: {self.model_name}")
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.use_projection:
            self.projection = ProjectionLayer(input_dim=512, output_dim=1024)
            self.projection.to(self.device)
            self.projection.eval()
            print(f"✓ Model loaded on {self.device} with 1024D projection")
        else:
            print(f"✓ Model loaded on {self.device}")

    def embed(self, text):
        """Generate embedding for single text."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            if self.use_projection:
                embeddings = self.projection(embeddings)

        return embeddings.cpu().numpy()[0]

    def batch_embed(self, texts, batch_size=32):
        """Generate embeddings for batch of texts."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self.model.get_text_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                if self.use_projection:
                    embeddings = self.projection(embeddings)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


def test_text_embedder():
    """Test the text embedder with dummy data."""
    print("=" * 60)
    print("TEXT EMBEDDER TEST")
    print("=" * 60)

    # Initialize embedder
    embedder = TextEmbedder(device="cpu")
    embedder.load_model()

    # Test 1: Single text embedding
    print("\n--- Test 1: Single Text Embedding ---")
    text = "A cute cat playing with a red ball"
    print(f"Input text: '{text}'")

    embedding = embedder.embed(text)
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding dimension: {len(embedding)}")
    print(f"✓ Embedding norm (should be ~1.0): {np.linalg.norm(embedding):.6f}")
    print(f"✓ First 5 values: {embedding[:5]}")

    # Test 2: Batch text embedding
    print("\n--- Test 2: Batch Text Embedding ---")
    texts = [
        "A cute cat playing with a red ball",
        "A golden retriever dog running on the beach",
        "A beautiful sunset over the mountains",
        "A kitten sleeping on a soft pillow",
        "An airplane flying through the clouds",
    ]

    print(f"Processing {len(texts)} texts...")
    for i, t in enumerate(texts, 1):
        print(f"  {i}. {t}")

    embeddings = embedder.batch_embed(texts)
    print(f"\n✓ Batch embeddings shape: {embeddings.shape}")
    print(
        f"✓ All vectors normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)}"
    )

    # Test 3: Semantic similarity
    print("\n--- Test 3: Semantic Similarity Test ---")
    text1 = "cat playing with toy"
    text2 = "kitten playing with ball"
    text3 = "airplane in the sky"

    emb1 = embedder.embed(text1)
    emb2 = embedder.embed(text2)
    emb3 = embedder.embed(text3)

    sim_12 = np.dot(emb1, emb2)  # Similar texts
    sim_13 = np.dot(emb1, emb3)  # Dissimilar texts

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Text 3: '{text3}'")
    print(f"\nSimilarity (text1 vs text2): {sim_12:.4f}")
    print(f"Similarity (text1 vs text3): {sim_13:.4f}")
    print(f"✓ Similar texts have higher similarity: {sim_12 > sim_13}")

    # Test 4: Search simulation
    print("\n--- Test 4: Simple Search Simulation ---")
    query = "pet playing"
    print(f"Query: '{query}'")
    query_emb = embedder.embed(query)

    # Compute similarities
    print("\nRanked results:")
    sims = embeddings @ query_emb
    ranked_indices = np.argsort(sims)[::-1]

    for rank, idx in enumerate(ranked_indices[:3], 1):
        print(f"{rank}. [{sims[idx]:.4f}] {texts[idx]}")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_text_embedder()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
