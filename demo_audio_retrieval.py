"""
Audio Retrieval Demo - Interactive Example

Run this to test audio retrieval interactively.

Usage:
    python demo_audio_retrieval.py
"""

import os
import sys
from pathlib import Path
from src.retrieval import MultiModalSearchEngine


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def demo_audio_retrieval():
    """Interactive audio retrieval demonstration."""
    
    print_header("🎵 AUDIO RETRIEVAL DEMO")
    
    # Step 1: Get audio folder from user
    print("\n📂 Step 1: Locate Audio Files")
    print("-" * 70)
    
    default_folder = './test_data'
    folder = input(f"Enter audio folder path (default: {default_folder}): ").strip()
    
    if not folder:
        folder = default_folder
    
    if not os.path.exists(folder):
        print(f"\n❌ Folder not found: {folder}")
        print("\n💡 Please create the folder and add some audio files, then try again.")
        return 1
    
    # Find audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']:
        audio_files.extend(Path(folder).glob(ext))
    
    if not audio_files:
        print(f"\n❌ No audio files found in {folder}")
        print("\n💡 Supported formats: .wav, .mp3, .flac, .ogg, .m4a")
        return 1
    
    print(f"\n✓ Found {len(audio_files)} audio files:")
    for i, f in enumerate(audio_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   {i}. {f.name} ({size_mb:.2f} MB)")
    
    # Step 2: Initialize search engine
    print_header("⚙️  Step 2: Initialize Search Engine")
    
    print("\nLoading CLAP model (may take a moment)...")
    print("📥 This will download ~1.8GB model on first run")
    
    try:
        engine = MultiModalSearchEngine(
            embedding_dim=512,
            device='cpu',
            index_type='flat'
        )
        engine.initialize(modalities=['audio'])
        print("✓ Search engine initialized")
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        return 1
    
    # Step 3: Index audio files
    print_header("📚 Step 3: Index Audio Files")
    
    print(f"\nIndexing {len(audio_files)} files...")
    
    indexed = 0
    for i, audio_path in enumerate(audio_files, 1):
        try:
            print(f"   [{i}/{len(audio_files)}] {audio_path.name}...", end='')
            engine.ingest_content(
                str(audio_path),
                content_type='audio',
                metadata={
                    'filename': audio_path.name,
                    'path': str(audio_path),
                    'size_mb': audio_path.stat().st_size / (1024 * 1024)
                }
            )
            print(" ✓")
            indexed += 1
        except Exception as e:
            print(f" ❌ Error: {e}")
    
    print(f"\n✓ Successfully indexed {indexed}/{len(audio_files)} files")
    
    if indexed == 0:
        print("\n❌ No files were indexed. Exiting.")
        return 1
    
    # Step 4: Audio-to-Audio Search
    print_header("🔍 Step 4: Audio-to-Audio Search")
    
    print("\nSelect a query audio file:")
    for i, f in enumerate(audio_files, 1):
        print(f"   {i}. {f.name}")
    
    while True:
        try:
            choice = input(f"\nEnter number (1-{len(audio_files)}): ").strip()
            query_idx = int(choice) - 1
            if 0 <= query_idx < len(audio_files):
                break
            print(f"Please enter a number between 1 and {len(audio_files)}")
        except (ValueError, KeyboardInterrupt):
            print("\nAborted.")
            return 0
    
    query_audio = str(audio_files[query_idx])
    print(f"\n🎧 Searching for audio similar to: {audio_files[query_idx].name}")
    
    try:
        results = engine.search(
            query=query_audio,
            query_type='audio',
            k=min(5, len(audio_files)),
            filter_content_type='audio'
        )
        
        print(f"\n📊 Top {len(results)} Similar Audio Files:")
        print("-" * 70)
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            distance = result['distance']
            similarity_pct = 100 / (1 + distance)
            
            print(f"\n  {i}. {metadata['filename']}")
            print(f"     Similarity: {'█' * int(similarity_pct/5)}{' ' * (20-int(similarity_pct/5))} {similarity_pct:.1f}%")
            print(f"     Distance:   {distance:.4f}")
            
            if i == 1:
                is_exact = distance < 0.01
                if is_exact:
                    print(f"     ✓ EXACT MATCH (query file)")
                else:
                    print(f"     ⚠️  Not exact match (distance: {distance:.4f})")
        
        # Verify
        if results[0]['distance'] < 0.01:
            print("\n✅ TEST PASSED: Query audio is the top result")
        else:
            print("\n⚠️  Unexpected: Query audio is not the exact top result")
            
    except Exception as e:
        print(f"\n❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 5: Text-to-Audio Search
    print_header("💬 Step 5: Text-to-Audio Search")
    
    print("\nNow try searching audio using text descriptions!")
    print("\nExample queries:")
    print("   - 'music playing'")
    print("   - 'dog barking'")
    print("   - 'people talking'")
    print("   - 'nature sounds'")
    print("   - 'piano music'")
    
    while True:
        query_text = input("\n🔍 Enter text query (or 'q' to quit): ").strip()
        
        if query_text.lower() in ['q', 'quit', 'exit']:
            break
        
        if not query_text:
            continue
        
        try:
            print(f"\nSearching for: \"{query_text}\"...")
            text_embedding = engine.audio_embedder.embed_text(query_text)
            
            results = engine.vector_index.search(
                text_embedding,
                k=3,
                filter_fn=lambda m: m.get('content_type') == 'audio'
            )
            
            print(f"\n📊 Top 3 Results:")
            print("-" * 70)
            
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                distance = result['distance']
                similarity_pct = 100 / (1 + distance)
                
                print(f"\n  {i}. {metadata['filename']}")
                print(f"     Similarity: {'█' * int(similarity_pct/5)}{' ' * (20-int(similarity_pct/5))} {similarity_pct:.1f}%")
                print(f"     Distance:   {distance:.4f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    print_header("✅ DEMO COMPLETE")
    
    print("\n📊 Summary:")
    stats = engine.get_stats()
    print(f"   Total audio files indexed: {stats['num_vectors']}")
    print(f"   Embedding dimension: {stats['dimension']}")
    print(f"   Index type: {stats['index_type']}")
    
    print("\n💡 Key Takeaways:")
    print("   1. Audio-to-audio search finds similar audio files")
    print("   2. Text-to-audio search uses semantic understanding")
    print("   3. Lower distance = Higher similarity")
    print("   4. Query audio should be rank #1 with distance ≈ 0")
    
    print("\n🚀 Next Steps:")
    print("   - Test with your own audio files")
    print("   - Try different text descriptions")
    print("   - Integrate into your application")
    print("   - See AUDIO_RETRIEVAL_TESTING_GUIDE.md for more details")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(demo_audio_retrieval())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
