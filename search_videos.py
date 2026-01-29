"""
Search videos using text queries
Loads pre-built HNSW index for fast retrieval
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embedders.text_embedder import TextEmbedder
from src.database.vector_index import VectorIndex

# Index path
INDEX_PATH = "saved_indices/video_index"


def search_videos(query, k=5):
    """
    Search videos using text query

    Args:
        query: Text query (e.g., "a person walking")
        k: Number of results to return
    """
    print(f"\n{'='*80}")
    print(f"VIDEO SEARCH: '{query}'")
    print("="*80)

    # Load index
    print("\n1. Loading HNSW index...")
    vector_index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
    vector_index.load(INDEX_PATH)
    print(f"✓ Loaded index with {len(vector_index.metadata)} videos")

    # Initialize text embedder
    print("\n2. Initializing text embedder...")
    txt_embedder = TextEmbedder(device="cpu")
    txt_embedder.load_model()

    # Embed query
    print(f"\n3. Searching for: '{query}'")
    query_embedding = txt_embedder.embed(query)

    # Search
    results = vector_index.search(query_embedding, k=k)

    # Display results
    print(f"\nTop {k} results:\n")
    for rank, result in enumerate(results, 1):
        meta = result['metadata']
        similarity = result['similarity']

        print(f"Rank {rank}: {meta['video_name']}")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Path: {meta['video_path']}")
        print()

    txt_embedder.unload_model()
    return results


def main():
    

    if not os.path.exists(f"{INDEX_PATH}.index"):
        print(f"\n Index not found at {INDEX_PATH}")
        print("   Run: python build_all_indices.py")
        return

    example_queries = [
        "dancers in the street",
        "a cat playing with a toy"
    ]

    print("\nExample queries:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")

    for query in example_queries[:3]:  
        search_videos(query, k=5)
        print("\n" + "-"*80)

    print("   python -c \"from search_videos import search_videos; search_videos('your query here')\"")


if __name__ == "__main__":
    main()
