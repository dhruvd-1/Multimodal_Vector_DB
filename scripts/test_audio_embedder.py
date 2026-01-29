"""
Comprehensive test script for Audio Embedder (CLAP).
Tests cross-modal retrieval: Audio File <-> Text Descriptions.
"""

import os
import sys
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION: Set your audio file path here
# =============================================================================
AUDIO_FILE_PATH = os.path.join(os.path.dirname(__file__), "dogbarking.mp3")
# =============================================================================

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedders.audio_embedder import AudioEmbedder

def test_audio_cross_modal():
    """Test cross-modal similarity between audio and text."""
    print("\n--- Testing Cross-Modal Retrieval (Audio <-> Text) ---")
    
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"Skipping: Audio file not found at {AUDIO_FILE_PATH}")
        print("Please ensure the audio file exists in the scripts directory.")
        return

    try:
        # Check if we're on mobile or have limited resources
        embedder = AudioEmbedder(device='cpu')
        
        print(f"\n⚠️  CLAP model requires ~1.8GB download on first use")
        print(f"   Make sure you have sufficient disk space available.")
        print(f"   Audio file: {AUDIO_FILE_PATH}\n")
        
        embedder.load_model()
        
        # 1. Embed the Audio
        print(f"Embedding audio: {os.path.basename(AUDIO_FILE_PATH)}")
        audio_emb = embedder.embed(AUDIO_FILE_PATH)
        
        # 2. Embed the text queries
        # Note: Like CLIP, CLAP works better with descriptive phrases
        text_positive = "a dog barking loudly"
        text_negative = "someone playing piano music"
        
        print(f"Embedding text queries:")
        print(f"  Positive: '{text_positive}'")
        print(f"  Negative: '{text_negative}'")
        
        emb_pos = embedder.embed_text(text_positive)
        emb_neg = embedder.embed_text(text_negative)
        
        # 3. Calculate similarities
        sim_pos = np.dot(audio_emb, emb_pos)
        sim_neg = np.dot(audio_emb, emb_neg)
        
        print(f"\nResults:")
        print(f"  Similarity (Audio <-> '{text_positive}'): {sim_pos:.4f}")
        print(f"  Similarity (Audio <-> '{text_negative}'): {sim_neg:.4f}")
        
        # 4. Assert semantic alignment
        assert sim_pos > sim_neg, f"Cross-modal alignment failed! '{text_positive}' should be more similar than '{text_negative}'"
        print("\n✓ Cross-modal alignment passed! (Audio matches correct text)")
        
        embedder.unload_model()
        
    except OSError as e:
        if "No space left on device" in str(e):
            print(f"\n✗ ERROR: Disk space full!")
            print(f"   CLAP model requires ~1.8GB for download.")
            print(f"   Please free up disk space on your C: drive and try again.")
        else:
            print(f"Audio cross-modal test failed: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"Audio cross-modal test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run tests."""
    print("=" * 60)
    print("RUNNING AUDIO CROSS-MODAL TESTS")
    print("=" * 60)
    
    # Check if CLAP is installed first
    try:
        import laion_clap
        test_audio_cross_modal()
    except ImportError:
        print("\n✗ ERROR: laion-clap not installed.")
        print("Install it with: pip install laion-clap")
        return

    print("\n" + "=" * 60)
    print("TEST EXECUTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()