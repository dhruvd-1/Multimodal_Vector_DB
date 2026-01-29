"""
Unit tests for audio retrieval functionality.

Run with: pytest tests/test_audio_retrieval.py -v
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.retrieval import MultiModalSearchEngine
from src.embedders import AudioEmbedder


class TestAudioRetrieval:
    """Test suite for audio retrieval functionality."""
    
    @pytest.fixture
    def audio_engine(self):
        """Create search engine with audio support."""
        engine = MultiModalSearchEngine(
            embedding_dim=512,
            device='cpu',
            index_type='flat'
        )
        engine.initialize(modalities=['audio'])
        return engine
    
    @pytest.fixture
    def sample_audio_files(self, tmp_path):
        """
        Create or use sample audio files for testing.
        
        Note: In real tests, you'd need actual audio files.
        This is a placeholder that looks for test files.
        """
        # Try to find test audio files
        test_data_dir = Path('./test_data')
        
        if test_data_dir.exists():
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.flac']:
                audio_files.extend(test_data_dir.glob(ext))
            
            if audio_files:
                return [str(f) for f in audio_files[:3]]  # Use first 3 files
        
        # Skip test if no audio files available
        pytest.skip("No test audio files available. Create test_data/ with audio files.")
    
    def test_audio_embedder_initialization(self):
        """Test that audio embedder can be initialized."""
        embedder = AudioEmbedder(model_name='music_speech', device='cpu')
        
        assert embedder is not None
        assert embedder.embedding_dim == 512
        assert embedder.device == 'cpu'
    
    def test_audio_embedder_load(self):
        """Test loading the CLAP model."""
        embedder = AudioEmbedder(model_name='music_speech', device='cpu')
        embedder.load_model()
        
        assert embedder._is_loaded
        assert embedder.model is not None
    
    def test_audio_embedding_generation(self, sample_audio_files):
        """Test generating embeddings for audio files."""
        embedder = AudioEmbedder(device='cpu')
        embedder.load_model()
        
        audio_path = sample_audio_files[0]
        embedding = embedder.embed(audio_path)
        
        # Check embedding properties
        assert embedding is not None
        assert embedding.shape == (512,)
        assert np.isfinite(embedding).all()
        
        # Check normalization
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)
    
    def test_batch_audio_embedding(self, sample_audio_files):
        """Test batch embedding generation."""
        embedder = AudioEmbedder(device='cpu')
        embedder.load_model()
        
        embeddings = embedder.batch_embed(sample_audio_files)
        
        assert embeddings is not None
        assert embeddings.shape[0] == len(sample_audio_files)
        assert embeddings.shape[1] == 512
        assert np.isfinite(embeddings).all()
    
    def test_text_embedding_for_audio_search(self):
        """Test generating text embeddings in audio space."""
        embedder = AudioEmbedder(device='cpu')
        embedder.load_model()
        
        text_embedding = embedder.embed_text("dog barking")
        
        assert text_embedding is not None
        assert text_embedding.shape == (512,)
        assert np.isfinite(text_embedding).all()
        
        # Check normalization
        norm = np.linalg.norm(text_embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)
    
    def test_audio_ingestion(self, audio_engine, sample_audio_files):
        """Test ingesting audio files into search engine."""
        audio_path = sample_audio_files[0]
        
        audio_engine.ingest_content(
            audio_path,
            content_type='audio',
            metadata={'description': 'test audio', 'path': audio_path}
        )
        
        stats = audio_engine.get_stats()
        assert stats['num_vectors'] == 1
    
    def test_audio_to_audio_search(self, audio_engine, sample_audio_files):
        """Test audio-to-audio retrieval."""
        # Ingest audio files
        for i, audio_path in enumerate(sample_audio_files):
            audio_engine.ingest_content(
                audio_path,
                content_type='audio',
                metadata={'idx': i, 'path': audio_path}
            )
        
        # Search using first audio as query
        query_audio = sample_audio_files[0]
        results = audio_engine.search(
            query=query_audio,
            query_type='audio',
            k=3,
            filter_content_type='audio'
        )
        
        assert len(results) > 0
        assert len(results) <= 3
        
        # First result should be the query itself with very low distance
        assert results[0]['distance'] < 0.1  # Should be nearly 0
        assert results[0]['metadata']['path'] == query_audio
    
    def test_text_to_audio_search(self, audio_engine, sample_audio_files):
        """Test text-to-audio cross-modal search."""
        # Ingest audio files
        descriptions = ['dog barking', 'cat meowing', 'music playing']
        
        for audio_path, desc in zip(sample_audio_files, descriptions):
            audio_engine.ingest_content(
                audio_path,
                content_type='audio',
                metadata={'description': desc, 'path': audio_path}
            )
        
        # Search using text
        text_embedding = audio_engine.audio_embedder.embed_text("dog")
        results = audio_engine.vector_index.search(
            text_embedding,
            k=3,
            filter_fn=lambda m: m.get('content_type') == 'audio'
        )
        
        assert len(results) > 0
        assert all('metadata' in r for r in results)
        assert all('distance' in r for r in results)
    
    def test_audio_search_with_filtering(self, audio_engine, sample_audio_files):
        """Test audio search with metadata filtering."""
        # Ingest audio with categories
        categories = ['music', 'speech', 'nature']
        
        for audio_path, category in zip(sample_audio_files, categories):
            audio_engine.ingest_content(
                audio_path,
                content_type='audio',
                metadata={'category': category, 'path': audio_path}
            )
        
        # Search with filter
        query_audio = sample_audio_files[0]
        results = audio_engine.search(
            query=query_audio,
            query_type='audio',
            k=5,
            filter_content_type='audio'
        )
        
        # All results should be audio type
        assert all(r['metadata']['content_type'] == 'audio' for r in results)
    
    def test_embedding_consistency(self, sample_audio_files):
        """Test that embeddings are consistent across multiple calls."""
        embedder = AudioEmbedder(device='cpu')
        embedder.load_model()
        
        audio_path = sample_audio_files[0]
        
        # Generate embedding twice
        embedding1 = embedder.embed(audio_path)
        embedding2 = embedder.embed(audio_path)
        
        # Should be identical
        assert np.allclose(embedding1, embedding2, atol=1e-6)
    
    def test_different_audio_files_different_embeddings(self, sample_audio_files):
        """Test that different audio files produce different embeddings."""
        if len(sample_audio_files) < 2:
            pytest.skip("Need at least 2 audio files for this test")
        
        embedder = AudioEmbedder(device='cpu')
        embedder.load_model()
        
        embedding1 = embedder.embed(sample_audio_files[0])
        embedding2 = embedder.embed(sample_audio_files[1])
        
        # Embeddings should be different
        assert not np.allclose(embedding1, embedding2, atol=0.01)
        
        # But both should be normalized
        assert np.isclose(np.linalg.norm(embedding1), 1.0, atol=1e-5)
        assert np.isclose(np.linalg.norm(embedding2), 1.0, atol=1e-5)
    
    def test_search_results_ordering(self, audio_engine, sample_audio_files):
        """Test that search results are ordered by similarity."""
        # Ingest audio files
        for audio_path in sample_audio_files:
            audio_engine.ingest_content(
                audio_path,
                content_type='audio',
                metadata={'path': audio_path}
            )
        
        # Search
        query_audio = sample_audio_files[0]
        results = audio_engine.search(
            query=query_audio,
            query_type='audio',
            k=len(sample_audio_files)
        )
        
        # Results should be ordered by distance (ascending)
        distances = [r['distance'] for r in results]
        assert distances == sorted(distances)
    
    def test_k_parameter(self, audio_engine, sample_audio_files):
        """Test that k parameter limits number of results."""
        # Ingest audio files
        for audio_path in sample_audio_files:
            audio_engine.ingest_content(
                audio_path,
                content_type='audio',
                metadata={'path': audio_path}
            )
        
        # Search with k=2
        results = audio_engine.search(
            query=sample_audio_files[0],
            query_type='audio',
            k=2
        )
        
        assert len(results) == 2


class TestAudioEmbedderEdgeCases:
    """Test edge cases and error handling."""
    
    def test_embed_without_loading_model(self):
        """Test that embedding without loading model raises error."""
        embedder = AudioEmbedder(device='cpu')
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            embedder.embed('dummy.wav')
    
    def test_embed_nonexistent_file(self):
        """Test embedding non-existent audio file."""
        embedder = AudioEmbedder(device='cpu')
        embedder.load_model()
        
        with pytest.raises(Exception):  # Should raise some error
            embedder.embed('nonexistent_file.wav')
    
    def test_text_embed_without_loading(self):
        """Test text embedding without loading model."""
        embedder = AudioEmbedder(device='cpu')
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            embedder.embed_text("test")
    
    def test_unload_model(self):
        """Test model unloading."""
        embedder = AudioEmbedder(device='cpu')
        embedder.load_model()
        
        assert embedder._is_loaded
        
        embedder.unload_model()
        
        assert not embedder._is_loaded


# Markers for different test categories
pytest.mark.audio = pytest.mark.audio
pytest.mark.slow = pytest.mark.slow  # For tests that download models


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
