"""
Benchmark Cross-Modal HNSW Index
Tests quality and speed of unified cross-modal index vs individual modality indices
"""
import os
import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embedders.text_embedder import TextEmbedder
from src.embedders.audio_embedder import AudioEmbedder
from src.database.vector_index import VectorIndex


def benchmark_cross_modal_search():
    """
    Compare cross-modal unified index vs individual modality indices
    """
    print("\n" + "="*80)
    print("CROSS-MODAL HNSW BENCHMARK")
    print("="*80)
    print("\nThis benchmark compares:")
    print("  1. Unified cross-modal index (all modalities together)")
    print("  2. Individual modality indices (separate for each modality)")
    print("\nMetrics:")
    print("  - Search speed (latency)")
    print("  - Result quality (overlap with individual indices)")
    print("  - Cross-space contamination (wrong modality in top-K)")
    print("="*80)

    test_queries = [
        "dog playing in nature",
        "music and instruments",
        "space and astronomy",
        "people dancing",
        "water and ocean"
    ]

    print("\n1. Loading unified cross-modal index...")
    cross_modal_index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
    cross_modal_index.load("saved_indices/cross_modal_index")

    print("\n2. Loading individual modality indices...")
    individual_indices = {}

    for modality in ['image', 'video', 'audio', 'text']:
        index_path = f"saved_indices/{modality}_index"
        if os.path.exists(f"{index_path}.index"):
            idx = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
            idx.load(index_path)
            individual_indices[modality] = idx
            print(f"    Loaded {modality} index")
        else:
            print(f"     {modality} index not found (skipping)")

    print("\n3. Loading embedders...")
    clip_embedder = TextEmbedder(device="cpu")
    clip_embedder.load_model()

    clap_embedder = AudioEmbedder(model_name='music_speech', device="cpu", enable_fusion=True)
    clap_embedder.load_model()

    results_summary = []

    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)

    for query in test_queries:
        print(f"\n{''*80}")
        print(f"Query: '{query}'")
        print(f"{''*80}")

        clip_query = clip_embedder.embed(query)
        clap_query = clap_embedder.embed_text(query)

        print("\n[UNIFIED INDEX]")

        start_time = time.time()
        clip_results = cross_modal_index.search(clip_query, k=1000)
        clip_time = (time.time() - start_time) * 1000

        start_time = time.time()
        clap_results = cross_modal_index.search(clap_query, k=1000)
        clap_time = (time.time() - start_time) * 1000

        unified_grouped = defaultdict(list)
        for result in clip_results:
            modality = result['metadata']['modality']
            if modality in ['image', 'video', 'text']:
                unified_grouped[modality].append(result)

        for result in clap_results:
            modality = result['metadata']['modality']
            if modality == 'audio':
                unified_grouped[modality].append(result)

        print(f"  CLIP search time: {clip_time:.2f}ms")
        print(f"  CLAP search time: {clap_time:.2f}ms")
        print(f"  Total time: {clip_time + clap_time:.2f}ms")

        for modality in ['image', 'video', 'audio', 'text']:
            count = len(unified_grouped[modality])
            if count > 0:
                top_sim = unified_grouped[modality][0]['similarity']
                print(f"  {modality:8s}: {count:4d} results, top similarity: {top_sim:.4f}")

        if individual_indices:
            print("\n[INDIVIDUAL INDICES]")

            individual_grouped = defaultdict(list)
            total_individual_time = 0

            for modality, idx in individual_indices.items():
                query_emb = clap_query if modality == 'audio' else clip_query

                start_time = time.time()
                results = idx.search(query_emb, k=20)
                search_time = (time.time() - start_time) * 1000
                total_individual_time += search_time

                individual_grouped[modality] = results

                if len(results) > 0:
                    top_sim = results[0]['similarity']
                    print(f"  {modality:8s}: {len(results):4d} results, top similarity: {top_sim:.4f}, time: {search_time:.2f}ms")

            print(f"  Total time: {total_individual_time:.2f}ms")

            print("\n[OVERLAP ANALYSIS]")
            print("  Comparing top-20 results from unified vs individual indices:")

            for modality in ['image', 'video', 'audio', 'text']:
                if modality in individual_grouped and modality in unified_grouped:
                    unified_ids = set([r['metadata'].get('id', r['metadata'].get('filename', r['metadata'].get('title')))
                                      for r in unified_grouped[modality][:20]])
                    individual_ids = set([r['metadata'].get('id', r['metadata'].get('filename', r['metadata'].get('title')))
                                         for r in individual_grouped[modality][:20]])

                    overlap = len(unified_ids & individual_ids)
                    overlap_pct = (overlap / 20.0) * 100

                    print(f"  {modality:8s}: {overlap}/20 overlap ({overlap_pct:.1f}%)")

        print("\n[CONTAMINATION CHECK]")
        print("  Testing for wrong modality in CLIP/CLAP searches:")

        clip_audio_count = sum(1 for r in clip_results[:100] if r['metadata']['modality'] == 'audio')
        print(f"  CLIP search found {clip_audio_count}/100 audio results (should be ~0)")

        clap_non_audio = sum(1 for r in clap_results[:100] if r['metadata']['modality'] != 'audio')
        print(f"  CLAP search found {clap_non_audio}/100 non-audio results (should be ~0)")

        results_summary.append({
            'query': query,
            'unified_time_ms': clip_time + clap_time,
            'individual_time_ms': total_individual_time if individual_indices else 0,
            'clip_audio_contamination': clip_audio_count,
            'clap_contamination': clap_non_audio
        })

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    print("\n1. SPEED COMPARISON:")
    avg_unified = np.mean([r['unified_time_ms'] for r in results_summary])
    avg_individual = np.mean([r['individual_time_ms'] for r in results_summary if r['individual_time_ms'] > 0])

    print(f"   Unified index:     {avg_unified:.2f}ms average")
    if avg_individual > 0:
        print(f"   Individual indices: {avg_individual:.2f}ms average")
        speedup = avg_individual / avg_unified
        print(f"   Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    print("\n2. CONTAMINATION (cross-space leakage):")
    avg_clip_audio = np.mean([r['clip_audio_contamination'] for r in results_summary])
    avg_clap_other = np.mean([r['clap_contamination'] for r in results_summary])
    print(f"   CLIP→Audio contamination:  {avg_clip_audio:.1f}/100 (lower is better)")
    print(f"   CLAP→Other contamination:  {avg_clap_other:.1f}/100 (lower is better)")

    print("\n3. KEY FINDINGS:")
    if avg_clip_audio < 5:
        print("    CLIP/CLAP spaces are well-separated (low contamination)")
    else:
        print("     Moderate cross-space contamination detected")

    if avg_unified < avg_individual * 1.5:
        print("    Unified index is competitive in speed")
    else:
        print("     Unified index is slower (but searches all modalities)")

    print("\n4. RECOMMENDATION:")
    print("   The unified cross-modal index is useful for:")
    print("   • Exploratory search across all modalities")
    print("   • When you need results from multiple modalities")
    print("   • Initial ranking before re-ranking")
    print("\n   Use individual indices when:")
    print("   • You know the target modality")
    print("   • Maximum precision is required")
    print("   • Speed is critical")

    clip_embedder.unload_model()
    clap_embedder.unload_model()

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    try:
        benchmark_cross_modal_search()
    except Exception as e:
        print(f"\n Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
