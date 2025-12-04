import pytest
import numpy as np
from PIL import Image
import tempfile
import os

from src.embedders import TextEmbedder, ImageEmbedder


class TestTextEmbedder:
    """Test suite for TextEmbedder."""

    @pytest.fixture
    def text_embedder(self):
        """Create text embedder instance."""
        embedder = TextEmbedder(device='cpu')
        embedder.load_model()
        return embedder

    def test_single_embed(self, text_embedder):
        """Test single text embedding."""
        text = "A photo of a cat"
        embedding = text_embedder.embed(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)
        assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-5  # Normalized

    def test_batch_embed(self, text_embedder):
        """Test batch text embedding."""
        texts = [
            "A photo of a cat",
            "A photo of a dog",
            "A beautiful sunset"
        ]

        embeddings = text_embedder.batch_embed(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 512)

        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_semantic_similarity(self, text_embedder):
        """Test that similar texts have higher similarity."""
        text1 = "cat playing with ball"
        text2 = "kitten playing with toy"
        text3 = "airplane flying in sky"

        emb1 = text_embedder.embed(text1)
        emb2 = text_embedder.embed(text2)
        emb3 = text_embedder.embed(text3)

        # Similarity: cat-kitten should be higher than cat-airplane
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)

        assert sim_12 > sim_13


class TestImageEmbedder:
    """Test suite for ImageEmbedder."""

    @pytest.fixture
    def image_embedder(self):
        """Create image embedder instance."""
        embedder = ImageEmbedder(device='cpu')
        embedder.load_model()
        return embedder

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        return img

    def test_single_embed_pil(self, image_embedder, test_image):
        """Test single image embedding with PIL image."""
        embedding = image_embedder.embed(test_image)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)
        assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-5

    def test_single_embed_path(self, image_embedder, test_image):
        """Test single image embedding with file path."""
        # Save test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            test_image.save(f.name)
            temp_path = f.name

        try:
            embedding = image_embedder.embed(temp_path)

            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (512,)

        finally:
            os.unlink(temp_path)

    def test_batch_embed(self, image_embedder):
        """Test batch image embedding."""
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green'),
            Image.new('RGB', (224, 224), color='blue'),
        ]

        embeddings = image_embedder.batch_embed(images)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 512)

        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
