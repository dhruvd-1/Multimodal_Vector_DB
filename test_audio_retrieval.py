"""
Audio Retrieval Test Script

Tests the audio search functionality:
1. Ingests audio files into the database
2. Performs audio-to-audio search (find similar audio)
3. Performs text-to-audio search (search audio using text description)
4. Verifies the correct audio files are retrieved

Usage:
    python test_audio_retrieval.py
"""

import os
import sys
from pathlib import Path
import numpy as np

from src.retrieval import MultiModalSearchEngine
from src.embedders import AudioEmbedder


def create_test_audio_db(audio_files, storage_path='./test_audio_db'):
    """
    Create a test audio database by ingesting audio files.
    
    Args:
        audio_files: List of tuples (audio_path, description, metadata)
        storage_path: Path to store the database
        
    Returns:
        Initialized search engine with audio files indexed
    """
    print("=" * 70)
    print("🎵 AUDIO RETRIEVAL TEST")
    print("=" * 70)
    
    # Initialize search engine with audio support
    engine = MultiModalSearchEngine(
        embedding_dim=512,
        device='cpu',
        index_type='flat',  # Use flat for exact search
        storage_path=storage_path
    )
    
    print("\n📥 Initializing audio embedder...")
    engine.initialize(modalities=['audio', 'text'])
    
    print("\n📚 Ingesting audio files into database...")
    for i, (audio_path, description, metadata) in enumerate(audio_files):
        if not os.path.exists(audio_path):
            print(f"⚠️  Warning: {audio_path} not found, skipping...")
            continue
            
        print(f"  [{i+1}/{len(audio_files)}] {Path(audio_path).name}: {description}")
        
        # Add metadata
        meta = metadata.copy() if metadata else {}
        meta['description'] = description
        meta['filename'] = os.path.basename(audio_path)
        meta['audio_path'] = audio_path
        
        try:
            engine.ingest_content(
                audio_path,
                content_type='audio',
                metadata=meta
            )
        except Exception as e:
            print(f"    ❌ Error ingesting {audio_path}: {e}")
            continue
    
    print(f"\n✓ Database created with {engine.vector_index.get_stats()['num_vectors']} audio files")
    return engine


def test_audio_to_audio_search(engine, query_audio_path, k=5):
    """
    Test audio-to-audio search: find similar audio files.
    
    Args:
        engine: Initialized search engine
        query_audio_path: Path to query audio file
        k: Number of results to return
        
    Returns:
        List of search results
    """
    print("\n" + "=" * 70)
    print("🔍 TEST 1: AUDIO-TO-AUDIO SEARCH (Find Similar Audio)")
    print("=" * 70)
    
    if not os.path.exists(query_audio_path):
        print(f"❌ Query audio not found: {query_audio_path}")
        return None
    
    print(f"\n🎧 Query: {Path(query_audio_path).name}")
    
    try:
        results = engine.search(
            query=query_audio_path,
            query_type='audio',
            k=k,
            filter_content_type='audio'
        )
        
        print(f"\n📊 Top {k} Results:")
        print("-" * 70)
        
        for i, result in enumerate(results):
            metadata = result['metadata']
            distance = result['distance']
            similarity = 1 / (1 + distance)  # Convert distance to similarity score
            
            print(f"\n  Rank {i+1}:")
            print(f"    File: {metadata.get('filename', 'N/A')}")
            print(f"    Description: {metadata.get('description', 'N/A')}")
            print(f"    Distance: {distance:.4f}")
            print(f"    Similarity: {similarity:.4f}")
            
            # Check if this is the query file itself (should be rank 1 with distance ~0)
            if i == 0:
                is_same = metadata.get('audio_path') == query_audio_path
                print(f"    ✓ Exact match: {'YES' if is_same and distance < 0.01 else 'NO'}")
        
        return results
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_text_to_audio_search(engine, query_text, k=5):
    """
    Test text-to-audio search: search audio using text description.
    
    Args:
        engine: Initialized search engine
        query_text: Text query (e.g., "dog barking")
        k: Number of results to return
        
    Returns:
        List of search results
    """
    print("\n" + "=" * 70)
    print("🔍 TEST 2: TEXT-TO-AUDIO SEARCH (Find Audio Using Text)")
    print("=" * 70)
    
    print(f"\n💬 Query: \"{query_text}\"")
    
    try:
        # Generate text embedding using audio embedder
        if engine.audio_embedder is None:
            print("❌ Audio embedder not initialized")
            return None
        
        # Use the audio embedder's text embedding capability
        text_embedding = engine.audio_embedder.embed_text(query_text)
        
        # Search using the text embedding
        results = engine.vector_index.search(
            text_embedding,
            k=k,
            filter_fn=lambda meta: meta.get('content_type') == 'audio'
        )
        
        print(f"\n📊 Top {k} Results:")
        print("-" * 70)
        
        for i, result in enumerate(results):
            metadata = result['metadata']
            distance = result['distance']
            similarity = 1 / (1 + distance)
            
            print(f"\n  Rank {i+1}:")
            print(f"    File: {metadata.get('filename', 'N/A')}")
            print(f"    Description: {metadata.get('description', 'N/A')}")
            print(f"    Distance: {distance:.4f}")
            print(f"    Similarity: {similarity:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_results(results, expected_keywords, top_k=3):
    """
    Verify that search results contain expected keywords.
    
    Args:
        results: Search results
        expected_keywords: List of keywords to look for
        top_k: Check only top K results
        
    Returns:
        True if keywords found in top results
    """
    print("\n" + "=" * 70)
    print("✅ VERIFICATION")
    print("=" * 70)
    
    if not results:
        print("❌ No results to verify")
        return False
    
    print(f"\n🔎 Looking for keywords: {expected_keywords}")
    print(f"   In top {top_k} results")
    
    found = False
    for i, result in enumerate(results[:top_k]):
        description = result['metadata'].get('description', '').lower()
        filename = result['metadata'].get('filename', '').lower()
        combined = f"{description} {filename}"
        
        for keyword in expected_keywords:
            if keyword.lower() in combined:
                print(f"\n✓ Found '{keyword}' in result {i+1}")
                print(f"  {result['metadata'].get('filename', 'N/A')}")
                found = True
                break
    
    if found:
        print("\n✅ VERIFICATION PASSED: Expected audio found in results")
    else:
        print("\n⚠️  VERIFICATION WARNING: Expected keywords not found in top results")
    
    return found


def main():
    """Main test function."""
    
    # Example test data - modify these paths to your actual audio files
    print("\n📝 Setting up test audio files...")
    print("   (Modify the audio_files list with your actual audio file paths)\n")
    
    # TODO: Replace these with your actual audio files
    audio_files = [
        # Format: (path, description, metadata)
        ("./test_data/dog_bark.wav", "dog barking loudly", {"category": "animals"}),
        ("./test_data/cat_meow.wav", "cat meowing", {"category": "animals"}),
        ("./test_data/piano_music.wav", "classical piano music", {"category": "music"}),
        ("./test_data/guitar_solo.wav", "electric guitar solo", {"category": "music"}),
        ("./test_data/ocean_waves.wav", "ocean waves crashing", {"category": "nature"}),
    ]
    
    # Check if example files exist, if not, provide instructions
    if not any(os.path.exists(path) for path, _, _ in audio_files):
        print("⚠️  No audio files found at the specified paths!")
        print("\n📋 Instructions:")
        print("   1. Create a test_data/ directory")
        print("   2. Add some audio files (.wav, .mp3, etc.)")
        print("   3. Update the audio_files list in this script with your file paths")
        print("\n   Or provide audio file paths when running the script:")
        print("   Example structure:")
        print("     test_data/")
        print("       ├── dog_bark.wav")
        print("       ├── cat_meow.wav")
        print("       └── piano_music.wav")
        
        # Ask user if they want to continue with custom paths
        print("\n" + "=" * 70)
        response = input("Do you have audio files ready? (y/n): ").lower()
        if response != 'y':
            print("\n👋 Please prepare audio files and try again!")
            return
        
        # Get custom audio files from user
        print("\n📂 Enter your audio file paths (press Enter when done):")
        audio_files = []
        i = 1
        while True:
            path = input(f"   Audio file {i} path (or Enter to finish): ").strip()
            if not path:
                break
            if not os.path.exists(path):
                print(f"   ⚠️  File not found: {path}")
                continue
            description = input(f"   Description for {Path(path).name}: ").strip()
            audio_files.append((path, description, {}))
            i += 1
        
        if not audio_files:
            print("\n❌ No audio files provided. Exiting.")
            return
    
    try:
        # Create database
        engine = create_test_audio_db(audio_files)
        
        # Get stats
        stats = engine.get_stats()
        print(f"\n📊 Database Statistics:")
        print(f"   Total vectors: {stats['num_vectors']}")
        print(f"   Dimension: {stats['dimension']}")
        print(f"   Index type: {stats['index_type']}")
        
        if stats['num_vectors'] == 0:
            print("\n❌ No audio files were successfully ingested. Exiting.")
            return
        
        # Test 1: Audio-to-audio search
        if audio_files:
            query_audio = audio_files[0][0]  # Use first audio as query
            results_audio = test_audio_to_audio_search(engine, query_audio, k=5)
            
            if results_audio:
                verify_results(
                    results_audio,
                    expected_keywords=[audio_files[0][1].split()[0]],  # First word of description
                    top_k=1  # The exact match should be rank 1
                )
        
        # Test 2: Text-to-audio search
        text_queries = [
            "dog barking",
            "music playing",
            "nature sounds"
        ]
        
        print("\n" + "=" * 70)
        print("Running multiple text queries...")
        print("=" * 70)
        
        for query in text_queries:
            results_text = test_text_to_audio_search(engine, query, k=3)
            if results_text:
                # Optional: verify each query
                pass
        
        print("\n" + "=" * 70)
        print("✅ AUDIO RETRIEVAL TEST COMPLETED")
        print("=" * 70)
        print("\n💡 Tips:")
        print("   - The first result in audio-to-audio search should be the query itself")
        print("   - Text-to-audio search relies on CLAP's cross-modal understanding")
        print("   - Lower distance values indicate higher similarity")
        print("   - Check if the top results match your expectations")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
