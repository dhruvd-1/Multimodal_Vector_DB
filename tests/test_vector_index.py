import pytest
import numpy as np
import tempfile
import os

from src.database import VectorIndex


class TestVectorIndex:
    """Test suite for VectorIndex."""

    @pytest.fixture
    def dimension(self):
        """Embedding dimension for tests."""
        return 128

    @pytest.fixture
    def sample_vectors(self, dimension):
        """Generate sample vectors."""
        np.random.seed(42)
        return np.random.randn(100, dimension).astype('float32')

    @pytest.fixture
    def sample_metadata(self):
        """Generate sample metadata."""
        return [{'idx': i, 'type': 'test'} for i in range(100)]

    def test_flat_index(self, dimension, sample_vectors, sample_metadata):
        """Test flat index (exact search)."""
        index = VectorIndex(dimension=dimension, index_type='flat')
        index.build_index()
        index.add_vectors(sample_vectors, sample_metadata)

        # Test search
        query = sample_vectors[0]
        results = index.search(query, k=5)

        assert len(results) == 5
        assert results[0]['metadata']['idx'] == 0  # Should find itself
        assert results[0]['distance'] < 1e-5  # Near-zero distance

    def test_ivf_index(self, dimension, sample_vectors, sample_metadata):
        """Test IVF index (approximate search)."""
        index = VectorIndex(dimension=dimension, index_type='ivf')
        index.build_index(nlist=10)
        index.add_vectors(sample_vectors, sample_metadata)

        # Test search
        query = sample_vectors[0]
        results = index.search(query, k=5)

        assert len(results) == 5
        # IVF might not be exact but should find close matches
        assert results[0]['distance'] < 1.0

    def test_hnsw_index(self, dimension, sample_vectors, sample_metadata):
        """Test HNSW index (graph-based search)."""
        index = VectorIndex(dimension=dimension, index_type='hnsw')
        index.build_index(hnsw_m=16)
        index.add_vectors(sample_vectors, sample_metadata)

        # Test search
        query = sample_vectors[0]
        results = index.search(query, k=5)

        assert len(results) == 5
        assert results[0]['metadata']['idx'] == 0

    def test_batch_search(self, dimension, sample_vectors, sample_metadata):
        """Test batch search."""
        index = VectorIndex(dimension=dimension, index_type='flat')
        index.build_index()
        index.add_vectors(sample_vectors, sample_metadata)

        # Batch search
        queries = sample_vectors[:5]
        all_results = index.search_batch(queries, k=3)

        assert len(all_results) == 5
        for results in all_results:
            assert len(results) == 3

    def test_save_load(self, dimension, sample_vectors, sample_metadata):
        """Test saving and loading index."""
        index = VectorIndex(dimension=dimension, index_type='flat')
        index.build_index()
        index.add_vectors(sample_vectors, sample_metadata)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_index')
            index.save(path)

            # Load
            index2 = VectorIndex(dimension=dimension)
            index2.load(path)

            # Test loaded index
            query = sample_vectors[0]
            results = index2.search(query, k=5)

            assert len(results) == 5
            assert results[0]['metadata']['idx'] == 0

    def test_filter_search(self, dimension, sample_vectors, sample_metadata):
        """Test search with filtering."""
        # Add varied metadata
        for i, meta in enumerate(sample_metadata):
            meta['category'] = 'A' if i < 50 else 'B'

        index = VectorIndex(dimension=dimension, index_type='flat')
        index.build_index()
        index.add_vectors(sample_vectors, sample_metadata)

        # Search with filter
        query = sample_vectors[0]
        filter_fn = lambda meta: meta.get('category') == 'A'

        results = index.search(query, k=5, filter_fn=filter_fn)

        assert len(results) == 5
        for result in results:
            assert result['metadata']['category'] == 'A'

    def test_get_stats(self, dimension, sample_vectors, sample_metadata):
        """Test getting index statistics."""
        index = VectorIndex(dimension=dimension, index_type='flat')
        index.build_index()
        index.add_vectors(sample_vectors, sample_metadata)

        stats = index.get_stats()

        assert stats['total_vectors'] == 100
        assert stats['dimension'] == dimension
        assert stats['index_type'] == 'flat'
