"""
Quick Audio Retrieval Test - Simple Example

This is a minimal example showing how to:
1. Index audio files
2. Search for similar audio
3. Verify results

Quick start:
    python test_audio_quick.py
"""

import os
from pathlib import Path
from src.retrieval import MultiModalSearchEngine


def quick_audio_test(audio_folder='./test_data'):
    """
    Quick test of audio retrieval.
    
    Args:
        audio_folder: Folder containing audio files
    """
    print("🎵 Quick Audio Retrieval Test\n")
    
    # Find audio files in folder
    if not os.path.exists(audio_folder):
        print(f"❌ Folder not found: {audio_folder}")
        print("Please create a 'test_data' folder with some audio files (.wav, .mp3, etc.)")
        return
    
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
        audio_files.extend(Path(audio_folder).glob(ext))
    
    if not audio_files:
        print(f"❌ No audio files found in {audio_folder}")
        print("Please add some audio files (.wav, .mp3, .flac, .ogg)")
        return
    
    print(f"📂 Found {len(audio_files)} audio files:")
    for f in audio_files:
        print(f"   - {f.name}")
    
    # Initialize search engine
    print("\n📥 Initializing audio search engine...")
    engine = MultiModalSearchEngine(
        embedding_dim=512,
        device='cpu',
        index_type='flat'
    )
    engine.initialize(modalities=['audio'])
    
    # Ingest audio files
    print("\n📚 Indexing audio files...")
    for i, audio_path in enumerate(audio_files):
        print(f"   [{i+1}/{len(audio_files)}] {audio_path.name}")
        engine.ingest_content(
            str(audio_path),
            content_type='audio',
            metadata={'filename': audio_path.name, 'path': str(audio_path)}
        )
    
    print(f"\n✓ Indexed {len(audio_files)} files")
    
    # Search using first audio as query
    query_audio = str(audio_files[0])
    print(f"\n🔍 Searching for similar audio to: {audio_files[0].name}")
    
    results = engine.search(
        query=query_audio,
        query_type='audio',
        k=min(5, len(audio_files)),
        filter_content_type='audio'
    )
    
    # Display results
    print(f"\n📊 Results:")
    print("-" * 60)
    for i, result in enumerate(results):
        metadata = result['metadata']
        distance = result['distance']
        similarity_pct = 100 / (1 + distance)
        
        print(f"\n{i+1}. {metadata['filename']}")
        print(f"   Similarity: {similarity_pct:.1f}%")
        print(f"   Distance: {distance:.4f}")
        
        # Check if it's the query itself
        if i == 0:
            is_exact = distance < 0.01
            status = "✓ EXACT MATCH" if is_exact else "⚠️ Not exact match"
            print(f"   {status}")
    
    # Verify
    print("\n" + "=" * 60)
    if results and results[0]['distance'] < 0.01:
        print("✅ TEST PASSED: Query audio found as top result")
    else:
        print("⚠️  TEST WARNING: Expected query audio as top result")
    
    # Text-to-audio search (bonus)
    print("\n" + "=" * 60)
    print("💬 Bonus: Text-to-Audio Search")
    print("=" * 60)
    
    test_queries = ["music playing", "people talking", "animal sounds"]
    print("\nTry these text queries:")
    for query in test_queries:
        print(f"   - \"{query}\"")
    
    # Example text search
    if engine.audio_embedder:
        print(f"\n🔍 Example: Searching for 'music'...")
        try:
            text_embedding = engine.audio_embedder.embed_text("music")
            results = engine.vector_index.search(
                text_embedding,
                k=3,
                filter_fn=lambda m: m.get('content_type') == 'audio'
            )
            print("\n📊 Top 3 results:")
            for i, result in enumerate(results):
                print(f"   {i+1}. {result['metadata']['filename']}")
        except Exception as e:
            print(f"   Note: {e}")


if __name__ == '__main__':
    import sys
    
    # Allow custom folder path
    folder = sys.argv[1] if len(sys.argv) > 1 else './test_data'
    
    try:
        quick_audio_test(folder)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
