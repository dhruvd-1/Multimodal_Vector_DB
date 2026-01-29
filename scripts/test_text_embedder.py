"""
Comprehensive test script for Text Embedder and Matryoshka Embeddings.
"""

import os
import sys
import numpy as np
import pytest
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedders.text_embedder import TextEmbedder
from src.embedders.projection import MatryoshkaProjection

def test_text_embedder_loading():
    """Test loading and unloading of the text embedder."""
    print("\n--- Testing Model Loading ---")
    embedder = TextEmbedder(device='cpu')
    assert not embedder.is_loaded
    
    embedder.load_model()
    assert embedder.is_loaded
    assert embedder.embedding_dim == 512
    
    embedder.unload_model()
    assert not embedder.is_loaded
    print("✓ Model loading/unloading passed")

def test_basic_embedding():
    """Test basic text embedding generation."""
    print("\n--- Testing Basic Embedding ---")
    embedder = TextEmbedder(device='cpu')
    embedder.load_model()
    
    text = "Hello world"
    embedding = embedder.embed(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    # Check normalization
    assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
    
    print("✓ Basic embedding passed")
    embedder.unload_model()

def test_batch_embedding():
    """Test batch embedding generation."""
    print("\n--- Testing Batch Embedding ---")
    embedder = TextEmbedder(device='cpu')
    embedder.load_model()
    
    texts = ["Hello world", "Another sentence", "Testing batch"]
    embeddings = embedder.batch_embed(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 512)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    
    print("✓ Batch embedding passed")
    embedder.unload_model()

def test_matryoshka_embedding():
    """Test Matryoshka variable dimension embeddings with TRAINED WEIGHTS."""
    print("\n--- Testing Matryoshka Embedding (Trained Weights) ---")
    
    weights_path = os.path.join("models", "matryoshka_final.pt")
    
    # Initialize with Matryoshka enabled
    embedder = TextEmbedder(
        device='cpu',
        use_matryoshka=True,
        matryoshka_dims=[512, 256, 128, 64]
    )
    embedder.load_model()
    
    # Load trained weights if they verify existence
    if os.path.exists(weights_path):
        print(f"Loading trained weights from: {weights_path}")
        embedder.load_matryoshka_weights(weights_path)
    else:
        print(f"Warning: {weights_path} not found! Testing with random init.")
    
    text = "A test sentence for variable dimensions"
    
    # Test all supported dimensions
    for dim in [512, 256, 128, 64]:
        emb = embedder.embed(text, output_dim=dim)
        assert emb.shape == (dim,)
        assert np.allclose(np.linalg.norm(emb), 1.0, atol=1e-5)
        print(f"✓ {dim}D embedding passed")
        
    # Test Semantic Preservation at Low Dimension (64D)
    print("\nVerifying Semantic Quality at 64D:")
    vec_anchor = embedder.embed("cat", output_dim=64)
    vec_pos = embedder.embed("kitten", output_dim=64)
    vec_neg = embedder.embed("car", output_dim=64)
    
    sim_pos = np.dot(vec_anchor, vec_pos)
    sim_neg = np.dot(vec_anchor, vec_neg)
    
    print(f"  64D Sim 'cat' vs 'kitten': {sim_pos:.4f}")
    print(f"  64D Sim 'cat' vs 'car':    {sim_neg:.4f}")
    
    if os.path.exists(weights_path):
        assert sim_pos > sim_neg, "Trained 64D model lost semantic meaning!"
        print("✓ Trained model preserves semantics at low dimension")
    
    # Test multi-scale retrieval
    multi_scale = embedder.embed_multi_scale(text)
    assert len(multi_scale) == 4
    print("✓ Multi-scale embedding passed")
    
    embedder.unload_model()

def test_semantic_similarity():
    """
    Test that the model captures semantic meaning.
    
    Note: CLIP models are trained on image-caption pairs and work best with 
    descriptive phrases (e.g., "a photo of a cat") rather than single words.
    Single-word embeddings may yield unexpected similarities.
    """
    print("\n--- Testing Semantic Similarity ---")
    embedder = TextEmbedder(device='cpu')
    embedder.load_model()
    
    # 1. Consistency Check
    text = "consistent input"
    emb1 = embedder.embed(text)
    emb2 = embedder.embed(text)
    
    # Ensure exact same input gives exact same output
    assert np.allclose(emb1, emb2), "Embeddings for same input are not identical!"
    print("✓ Consistency check passed (Same input = Same output)")

    # 2. Semantic Check
    # CLIP works best with descriptive phrases, not single words
    # "a photo of a cat" should be closer to "a photo of a kitten" than "a photo of a car"
    vec_anchor = embedder.embed("a photo of a cat")
    vec_positive = embedder.embed("a photo of a kitten")
    vec_negative = embedder.embed("a photo of a car")
    
    # Calculate Cosine Similarity (vectors are already normalized)
    score_pos = np.dot(vec_anchor, vec_positive)
    score_neg = np.dot(vec_anchor, vec_negative)
    
    print(f"Similarity 'a photo of a cat' <-> 'a photo of a kitten': {score_pos:.4f}")
    print(f"Similarity 'a photo of a cat' <-> 'a photo of a car':    {score_neg:.4f}")
    
    assert score_pos > score_neg, "Model failed to group similar concepts!"
    print("✓ Semantic similarity passed")
    
    embedder.unload_model()

def main():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING TEXT EMBEDDER TESTS")
    print("=" * 60)
    
    try:
        test_text_embedder_loading()
        test_basic_embedding()
        test_batch_embedding()
        test_matryoshka_embedding()
        test_semantic_similarity()
        
        print("\n" + "=" * 60)
        print("ALL TEXT EMBEDDER TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
