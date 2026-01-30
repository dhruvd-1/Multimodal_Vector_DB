"""
Cross-Modal Search: Search across ALL modalities with a single query
Retrieves results from images, videos, audio, and text simultaneously
"""
import os
import sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embedders.text_embedder import TextEmbedder
from src.embedders.audio_embedder import AudioEmbedder
from src.database.vector_index import VectorIndex

INDEX_PATH = "saved_indices/cross_modal_index"


def search_cross_modal(query, k=20, group_by_modality=True):
    """
    Search across ALL modalities with a single query

    Args:
        query: Text query (e.g., "dog playing", "music", "space exploration")
        k: Number of total results to return
        group_by_modality: If True, group results by modality

    Returns:
        dict: Results grouped by modality or list of all results
    """
    print(f"\n{'='*80}")
    print(f"CROSS-MODAL SEARCH: '{query}'")
    print("="*80)

    print("\n1. Loading cross-modal HNSW index...")
    vector_index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
    vector_index.load(INDEX_PATH)

    total_items = len(vector_index.metadata)
    modality_counts = defaultdict(int)
    for meta in vector_index.metadata:
        modality_counts[meta['modality']] += 1

    print(f" Loaded index with {total_items:,} items:")
    for modality, count in sorted(modality_counts.items()):
        print(f"    {modality}: {count:,}")

    print("\n2. Initializing text embedder...")
    txt_embedder = TextEmbedder(device="cpu")
    txt_embedder.load_model()

    print(f"\n3. Searching across all modalities for: '{query}'")
    query_embedding = txt_embedder.embed(query)

    results = vector_index.search(query_embedding, k=k)

    if group_by_modality:
        grouped_results = defaultdict(list)
        for result in results:
            modality = result['metadata']['modality']
            grouped_results[modality].append(result)

        print(f"\n4. Top {k} results across all modalities:\n")
        print("="*80)

        for modality in ['image', 'video', 'audio', 'text']:
            if modality in grouped_results:
                modality_results = grouped_results[modality]
                print(f"\n {modality.upper()} ({len(modality_results)} results)")
                print("-"*80)

                for rank, result in enumerate(modality_results, 1):
                    meta = result['metadata']
                    similarity = result['similarity']

                    print(f"\n  Rank {rank}: {meta['display_name']}")
                    print(f"    Similarity: {similarity:.4f}")
                    print(f"    Preview: {meta['preview']}")

                    if modality == 'image' and 'captions' in meta:
                        print(f"    Captions: {meta['captions'][0][:100]}...")
                    elif modality == 'audio' and 'category' in meta:
                        print(f"    Category: {meta['category']}")

        print("\n" + "="*80)

        txt_embedder.unload_model()
        return grouped_results

    else:
        print(f"\n4. Top {k} results:\n")

        for rank, result in enumerate(results, 1):
            meta = result['metadata']
            similarity = result['similarity']
            modality = meta['modality']

            print(f"Rank {rank} [{modality.upper()}]: {meta['display_name']}")
            print(f"  Similarity: {similarity:.4f}")
            print(f"  Preview: {meta['preview']}")
            print()

        txt_embedder.unload_model()
        return results


def compare_modalities(query, k_per_modality=5):
    """
    Search and compare results across modalities

    Args:
        query: Text query
        k_per_modality: Number of results per modality
    """
    print(f"\n{'='*80}")
    print(f"CROSS-MODAL COMPARISON: '{query}'")
    print("="*80)

    print("\n1. Loading cross-modal HNSW index...")
    vector_index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
    vector_index.load(INDEX_PATH)

    print("\n2. Initializing embedders...")
    print("   - CLIP text encoder (for image/video/text)")
    clip_embedder = TextEmbedder(device="cpu")
    clip_embedder.load_model()

    print("   - CLAP text encoder (for audio)")
    clap_embedder = AudioEmbedder(model_name='music_speech', device="cpu", enable_fusion=True)
    clap_embedder.load_model()

    print(f"\n3. Searching for: '{query}'")
    clip_query_embedding = clip_embedder.embed(query)
    clap_query_embedding = clap_embedder.embed_text(query)

    print("\n4. Searching each modality...")
    grouped_results = defaultdict(list)

    print("   - Searching image/video/text with CLIP...")
    clip_results = vector_index.search(clip_query_embedding, k=10000)

    print("   - Searching audio with CLAP...")
    clap_results = vector_index.search(clap_query_embedding, k=10000)

    for result in clip_results:
        modality = result['metadata']['modality']
        if modality in ['image', 'video', 'text']:
            grouped_results[modality].append(result)

    for result in clap_results:
        modality = result['metadata']['modality']
        if modality == 'audio':
            grouped_results[modality].append(result)

    print("\n" + "="*80)
    print(f"TOP {k_per_modality} PER MODALITY")
    print("="*80)

    for modality in ['image', 'video', 'audio', 'text']:
        if modality in grouped_results:
            modality_results = grouped_results[modality][:k_per_modality]

            encoder = "CLAP" if modality == "audio" else "CLIP"
            print(f"\n{modality.upper()} (using {encoder})")
            print("-"*80)

            for rank, result in enumerate(modality_results, 1):
                meta = result['metadata']
                similarity = result['similarity']
                print(f"  {rank}. {meta['display_name']} (similarity: {similarity:.4f})")

    clip_embedder.unload_model()
    clap_embedder.unload_model()


def main():
    
    if not os.path.exists(f"{INDEX_PATH}.index"):
        print(f"\n Index not found at {INDEX_PATH}")
        print("   Run: python build_cross_modal_index.py")
        return

    example_queries = [
        "dog",
        
    ]

    print("\nExample queries:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")

    print("\n" + "="*80)

    for query in example_queries[:3]:  # Run first 3 examples
        search_cross_modal(query, k=20, group_by_modality=True)
        print("\n\n" + "="*80)
        print("="*80 + "\n")

    
if __name__ == "__main__":
    main()
