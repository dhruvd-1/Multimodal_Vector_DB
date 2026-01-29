"""
Search Wikipedia articles using text queries
Loads pre-built HNSW index for fast retrieval
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embedders.text_embedder import TextEmbedder
from src.database.vector_index import VectorIndex

# Index path
INDEX_PATH = "saved_indices/text_index"


def search_text(query, k=5):
    """
    Search Wikipedia articles using text query

    Args:
        query: Text query (e.g., "quantum physics", "world war 2", "machine learning")
        k: Number of results to return
    """
    print(f"\n{'='*80}")
    print(f"TEXT SEARCH: '{query}'")
    print("="*80)

    # Load index
    print("\n1. Loading HNSW index...")
    vector_index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
    vector_index.load(INDEX_PATH)
    print(f"✓ Loaded index with {len(vector_index.metadata)} Wikipedia articles")

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
    print(f"\nTop {k} Wikipedia articles:\n")
    for rank, result in enumerate(results, 1):
        meta = result['metadata']
        similarity = result['similarity']

        print(f"Rank {rank}: {meta['title']}")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Preview: {meta['preview']}")
        print()

    txt_embedder.unload_model()
    return results


def main():
    """Interactive search"""
    print("\n🔍 WIKIPEDIA TEXT SEARCH (Simple English)")
    print("="*80)

    if not os.path.exists(f"{INDEX_PATH}.index"):
        print(f"\n❌ Index not found at {INDEX_PATH}")
        print("   Run: python build_text_index.py")
        return

    # Example queries
    example_queries = [
        "artificial intelligence and machine learning",
        "climate change and global warming",
        "ancient Rome and Roman Empire",
        "quantum physics and mechanics",
        "world war 2 history",
        "space exploration and astronomy",
        "human biology and anatomy",
        "computer programming languages"
    ]

    print("\nExample queries:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")

    print("\n" + "="*80)

    # Run example queries
    for query in example_queries[:5]:  # Run first 5 examples
        search_text(query, k=5)
        print("\n" + "-"*80)

    print("\n💡 To search with your own query, modify this script or run:")
    print("   python -c \"from search_text import search_text; search_text('your query here')\"")


if __name__ == "__main__":
    main()
