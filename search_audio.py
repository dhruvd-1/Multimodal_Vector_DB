"""
Search audio using text queries
Loads pre-built HNSW index for fast retrieval
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embedders.audio_embedder import AudioEmbedder
from src.database.vector_index import VectorIndex

# Index path
INDEX_PATH = "saved_indices/audio_index"


def search_audio(query, k=5):
    """
    Search audio using text query

    Args:
        query: Text query (e.g., "dog barking")
        k: Number of results to return
    """
   

    # Load index
    print("\n1. Loading HNSW index...")
    vector_index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
    vector_index.load(INDEX_PATH)
    print(f"✓ Loaded index with {len(vector_index.metadata)} audio files")

    # Initialize audio embedder (needed for text-to-audio)
    print("\n2. Initializing audio embedder...")
    audio_embedder = AudioEmbedder(model_name='music_speech', device='cpu', enable_fusion=True)
    audio_embedder.load_model()

    # Embed query (text-to-audio)
    print(f"\n3. Searching for: '{query}'")
    query_embedding = audio_embedder.embed_text(query)

    # Search
    results = vector_index.search(query_embedding, k=k)

    # Display results
    print(f"\nTop {k} results:\n")
    for rank, result in enumerate(results, 1):
        meta = result['metadata']
        similarity = result['similarity']

        print(f"Rank {rank}: {meta['filename']}")
        print(f"  Category: {meta['category']}")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Path: {meta['audio_path']}")
        print()

    audio_embedder.unload_model()
    return results


def main():
    

    if not os.path.exists(f"{INDEX_PATH}.index"):
        print(f"\n❌ Index not found at {INDEX_PATH}")
        print("   Run: python build_all_indices.py")
        return

    # Example queries
    example_queries = [
        "bird chirping in the morning",
    ]

    
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")

    print("\n" + "="*80)

    # Run example queries
    for query in example_queries[:5]:  # Run first 5 examples
        search_audio(query, k=5)
        print("\n" + "-"*80)

    print("   python -c \"from search_audio import search_audio; search_audio('your query here')\"")


if __name__ == "__main__":
    main()
