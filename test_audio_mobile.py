"""
Test audio embedder with mobile-optimized settings (smaller model).
This uses HTSAT-tiny instead of HTSAT-base to reduce memory/disk requirements.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedders.audio_embedder import AudioEmbedder

AUDIO_FILE_PATH = os.path.join(os.path.dirname(__file__), "dogbarking.mp3")

def test_audio_mobile():
    """Test audio embedder with mobile optimization."""
    print("\n--- Testing Audio Embedder (Mobile Mode) ---")
    
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"Audio file not found: {AUDIO_FILE_PATH}")
        return
    
    try:
        # Use mobile mode for smaller model
        embedder = AudioEmbedder(device='cpu')
        embedder._is_mobile = True  # Force mobile mode
        
        print("⚠️  Using mobile mode (HTSAT-tiny) to reduce memory usage")
        embedder.load_model()
        
        # Test audio embedding
        print(f"\nEmbedding audio: {os.path.basename(AUDIO_FILE_PATH)}")
        audio_emb = embedder.embed(AUDIO_FILE_PATH)
        
        print(f"Audio embedding shape: {audio_emb.shape}")
        print(f"Audio embedding norm: {np.linalg.norm(audio_emb):.4f}")
        
        # Test text embeddings  
        text1 = "a dog barking loudly"
        text2 = "someone playing piano music"
        
        print(f"\nEmbedding text:")
        print(f"  '{text1}'")
        print(f"  '{text2}'")
        
        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)
        
        # Calculate similarities
        sim1 = np.dot(audio_emb, emb1)
        sim2 = np.dot(audio_emb, emb2)
        
        print(f"\nResults:")
        print(f"  Similarity (Audio <-> '{text1}'): {sim1:.4f}")
        print(f"  Similarity (Audio <-> '{text2}'): {sim2:.4f}")
        
        if sim1 > sim2:
            print("\n✓ Cross-modal alignment passed! (Mobile mode)")
        else:
            print(f"\n✗ Expected '{text1}' to be more similar (got {sim1:.4f} vs {sim2:.4f})")
        
        embedder.unload_model()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("AUDIO EMBEDDER TEST (MOBILE MODE)")
    print("=" * 60)
    test_audio_mobile()
    print("\n" + "=" * 60)
