import pytest
import numpy as np
from PIL import Image
import tempfile
import os

from src.retrieval import MultiModalSearchEngine


class TestMultiModalSearchEngine:
    """Test suite for MultiModalSearchEngine."""

    @pytest.fixture
    def search_engine(self):
        """Create search engine instance."""
        engine = MultiModalSearchEngine(
            embedding_dim=512,
            device='cpu',
            index_type='flat'
        )
        engine.initialize(modalities=['text', 'image'])
        return engine

    def test_text_ingest_and_search(self, search_engine):
        """Test text ingestion and search."""
        # Ingest texts
        texts = [
            "A cute cat playing with a ball",
            "A dog running in the park",
            "A beautiful sunset over the ocean"
        ]

        for i, text in enumerate(texts):
            search_engine.ingest_content(
                text,
                content_type='text',
                metadata={'idx': i, 'text': text}
            )

        # Search
        results = search_engine.search(
            query="cat with toy",
            query_type='text',
            k=3
        )

        assert len(results) == 3
        # First result should be the cat text (most similar)
        assert 'cat' in results[0]['metadata']['text'].lower()

    def test_image_ingest_and_search(self, search_engine):
        """Test image ingestion and search."""
        # Create test images
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green'),
            Image.new('RGB', (224, 224), color='blue')
        ]

        # Ingest images
        for i, img in enumerate(images):
            search_engine.ingest_content(
                img,
                content_type='image',
                metadata={'idx': i, 'color': ['red', 'green', 'blue'][i]}
            )

        # Search with image
        query_img = images[0]
        results = search_engine.search(
            query=query_img,
            query_type='image',
            k=3
        )

        assert len(results) == 3
        # First result should be the same image
        assert results[0]['metadata']['color'] == 'red'
        assert results[0]['distance'] < 1e-5

    def test_cross_modal_search(self, search_engine):
        """Test cross-modal search (text query, image results)."""
        # Ingest images
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue')
        ]

        for i, img in enumerate(images):
            search_engine.ingest_content(
                img,
                content_type='image',
                metadata={'idx': i, 'type': 'image'}
            )

        # Ingest text
        search_engine.ingest_content(
            "A red apple",
            content_type='text',
            metadata={'idx': 2, 'type': 'text'}
        )

        # Search with text query
        results = search_engine.search(
            query="red color",
            query_type='text',
            k=3
        )

        assert len(results) == 3

    def test_filter_by_content_type(self, search_engine):
        """Test filtering results by content type."""
        # Ingest mixed content
        search_engine.ingest_content(
            "A cat",
            content_type='text',
            metadata={'title': 'text1'}
        )

        search_engine.ingest_content(
            Image.new('RGB', (224, 224), color='red'),
            content_type='image',
            metadata={'title': 'image1'}
        )

        # Search and filter for images only
        results = search_engine.search(
            query="cat",
            query_type='text',
            k=10,
            filter_content_type='image'
        )

        assert len(results) == 1
        assert results[0]['metadata']['content_type'] == 'image'

    def test_batch_ingest(self, search_engine):
        """Test batch ingestion."""
        texts = [
            "Text one",
            "Text two",
            "Text three"
        ]

        metadata_list = [
            {'idx': i} for i in range(len(texts))
        ]

        search_engine.batch_ingest(texts, 'text', metadata_list)

        # Verify ingestion
        stats = search_engine.get_stats()
        assert stats['total_vectors'] == 3

    def test_save_load(self, search_engine):
        """Test saving and loading search engine."""
        # Ingest some content
        search_engine.ingest_content(
            "Test text",
            content_type='text',
            metadata={'test': 'data'}
        )

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_engine')
            search_engine.save(path)

            # Create new engine and load
            engine2 = MultiModalSearchEngine(
                embedding_dim=512,
                device='cpu',
                index_type='flat'
            )
            engine2.initialize(modalities=['text', 'image'])
            engine2.load(path)

            # Verify
            stats = engine2.get_stats()
            assert stats['total_vectors'] == 1

    def test_get_stats(self, search_engine):
        """Test getting engine statistics."""
        stats = search_engine.get_stats()

        assert 'total_vectors' in stats
        assert 'enabled_embedders' in stats
        assert 'text' in stats['enabled_embedders']
        assert 'image' in stats['enabled_embedders']
