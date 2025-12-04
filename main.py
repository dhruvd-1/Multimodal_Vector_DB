"""
Multimodal Vector Database - Example Usage

This script demonstrates the complete pipeline for the multimodal vector database.
"""

from src.retrieval import MultiModalSearchEngine
from src.utils import EvaluationMetrics, Config
from PIL import Image
import numpy as np


def demo_text_search():
    """Demonstrate text-based search."""
    print("\n" + "=" * 60)
    print("DEMO 1: Text-Based Search")
    print("=" * 60)

    # Initialize search engine
    engine = MultiModalSearchEngine(
        embedding_dim=512,
        device='cpu',
        index_type='flat'
    )

    print("\nInitializing search engine...")
    engine.initialize(modalities=['text', 'image'])

    # Sample text documents
    documents = [
        "A cute cat playing with a red ball in the garden",
        "A golden retriever dog running on the beach",
        "A beautiful sunset over the mountains",
        "A kitten sleeping on a soft pillow",
        "An airplane flying through the clouds"
    ]

    print(f"\nIngesting {len(documents)} text documents...")
    for i, doc in enumerate(documents):
        engine.ingest_content(
            doc,
            content_type='text',
            metadata={'doc_id': i, 'text': doc}
        )

    # Perform search
    query = "cat playing with toy"
    print(f"\nSearching for: '{query}'")

    results, latency = EvaluationMetrics.measure_latency(
        engine.search,
        query=query,
        query_type='text',
        k=3
    )

    print(f"\nSearch completed in {latency:.2f}ms")
    print("\nTop 3 Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Distance: {result['distance']:.4f}")
        print(f"   Similarity: {result['similarity']:.4f}")
        print(f"   Text: {result['metadata']['text']}")


def demo_image_search():
    """Demonstrate image-based search."""
    print("\n" + "=" * 60)
    print("DEMO 2: Image-Based Search")
    print("=" * 60)

    # Initialize search engine
    engine = MultiModalSearchEngine(
        embedding_dim=512,
        device='cpu',
        index_type='flat'
    )

    print("\nInitializing search engine...")
    engine.initialize(modalities=['image'])

    # Create sample images (different colors)
    print("\nCreating and ingesting sample images...")
    images_data = [
        ('red', (255, 0, 0)),
        ('green', (0, 255, 0)),
        ('blue', (0, 0, 255)),
        ('yellow', (255, 255, 0)),
        ('purple', (128, 0, 128))
    ]

    for name, color in images_data:
        img = Image.new('RGB', (224, 224), color=color)
        engine.ingest_content(
            img,
            content_type='image',
            metadata={'name': name, 'color': color}
        )

    # Search with similar image
    print("\nSearching for images similar to red...")
    query_img = Image.new('RGB', (224, 224), color=(255, 0, 0))

    results = engine.search(
        query=query_img,
        query_type='image',
        k=3
    )

    print("\nTop 3 Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Distance: {result['distance']:.4f}")
        print(f"   Color: {result['metadata']['name']}")


def demo_cross_modal_search():
    """Demonstrate cross-modal search (text query -> image results)."""
    print("\n" + "=" * 60)
    print("DEMO 3: Cross-Modal Search (Text -> Images)")
    print("=" * 60)

    # Initialize search engine
    engine = MultiModalSearchEngine(
        embedding_dim=512,
        device='cpu',
        index_type='flat'
    )

    print("\nInitializing search engine...")
    engine.initialize(modalities=['text', 'image'])

    # Ingest images with descriptions
    print("\nIngesting images with descriptions...")
    image_descriptions = [
        ("A red apple", (255, 0, 0)),
        ("A green tree", (0, 255, 0)),
        ("A blue ocean", (0, 0, 255))
    ]

    for desc, color in image_descriptions:
        img = Image.new('RGB', (224, 224), color=color)
        engine.ingest_content(
            img,
            content_type='image',
            metadata={'description': desc}
        )

        # Also ingest the text
        engine.ingest_content(
            desc,
            content_type='text',
            metadata={'description': desc}
        )

    # Search with text, filter for images only
    query = "red fruit"
    print(f"\nSearching for: '{query}' (images only)")

    results = engine.search(
        query=query,
        query_type='text',
        k=3,
        filter_content_type='image'
    )

    print("\nTop Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Distance: {result['distance']:.4f}")
        print(f"   Description: {result['metadata']['description']}")
        print(f"   Type: {result['metadata']['content_type']}")


def demo_index_comparison():
    """Compare different index types."""
    print("\n" + "=" * 60)
    print("DEMO 4: Index Type Comparison")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    num_vectors = 1000
    dimension = 512

    print(f"\nGenerating {num_vectors} random vectors...")
    vectors = np.random.randn(num_vectors, dimension).astype('float32')
    metadata_list = [{'idx': i} for i in range(num_vectors)]

    # Test different index types
    index_types = ['flat', 'ivf', 'hnsw']

    for idx_type in index_types:
        print(f"\n--- Testing {idx_type.upper()} Index ---")

        engine = MultiModalSearchEngine(
            embedding_dim=dimension,
            device='cpu',
            index_type=idx_type
        )

        engine.initialize(modalities=['text'])

        # Measure ingestion time
        print(f"Adding {num_vectors} vectors...")
        _, ingest_time = EvaluationMetrics.measure_latency(
            engine.vector_index.add_vectors,
            vectors,
            metadata_list
        )

        print(f"Ingestion time: {ingest_time:.2f}ms")

        # Measure search time
        query = vectors[0]
        _, search_time = EvaluationMetrics.measure_latency(
            engine.vector_index.search,
            query,
            k=10
        )

        print(f"Search time: {search_time:.2f}ms")

        # Get stats
        stats = engine.get_stats()
        print(f"Total vectors: {stats['total_vectors']}")


def demo_batch_processing():
    """Demonstrate batch processing for efficiency."""
    print("\n" + "=" * 60)
    print("DEMO 5: Batch Processing")
    print("=" * 60)

    engine = MultiModalSearchEngine(
        embedding_dim=512,
        device='cpu',
        index_type='flat'
    )

    print("\nInitializing search engine...")
    engine.initialize(modalities=['text'])

    # Prepare batch data
    texts = [f"Document number {i}" for i in range(100)]
    metadata_list = [{'doc_id': i} for i in range(100)]

    # Individual ingestion
    print("\nIndividual ingestion (first 10)...")
    start_time = EvaluationMetrics.measure_latency.__func__
    individual_times = []

    for i in range(10):
        _, time_taken = EvaluationMetrics.measure_latency(
            engine.ingest_content,
            texts[i],
            'text',
            metadata_list[i]
        )
        individual_times.append(time_taken)

    avg_individual = np.mean(individual_times)
    print(f"Average time per document: {avg_individual:.2f}ms")

    # Batch ingestion
    print("\nBatch ingestion (remaining 90)...")
    _, batch_time = EvaluationMetrics.measure_latency(
        engine.batch_ingest,
        texts[10:],
        'text',
        metadata_list[10:]
    )

    avg_batch = batch_time / 90
    print(f"Average time per document: {avg_batch:.2f}ms")
    print(f"Speedup: {avg_individual / avg_batch:.2f}x")


def demo_evaluation_metrics():
    """Demonstrate evaluation metrics."""
    print("\n" + "=" * 60)
    print("DEMO 6: Evaluation Metrics")
    print("=" * 60)

    # Simulate retrieval results
    retrieved_ids = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    relevant_ids = [1, 2, 3, 10, 15, 20]

    print("\nRetrieved IDs:", retrieved_ids)
    print("Relevant IDs:", relevant_ids)

    # Calculate metrics
    recall_5 = EvaluationMetrics.calculate_recall_at_k(retrieved_ids, relevant_ids, k=5)
    recall_10 = EvaluationMetrics.calculate_recall_at_k(retrieved_ids, relevant_ids, k=10)

    precision_5 = EvaluationMetrics.calculate_precision_at_k(retrieved_ids, relevant_ids, k=5)
    precision_10 = EvaluationMetrics.calculate_precision_at_k(retrieved_ids, relevant_ids, k=10)

    ap = EvaluationMetrics.calculate_average_precision(retrieved_ids, relevant_ids)

    ndcg_5 = EvaluationMetrics.calculate_ndcg_at_k(retrieved_ids, relevant_ids, k=5)
    ndcg_10 = EvaluationMetrics.calculate_ndcg_at_k(retrieved_ids, relevant_ids, k=10)

    print("\nMetrics:")
    print(f"Recall@5:     {recall_5:.4f}")
    print(f"Recall@10:    {recall_10:.4f}")
    print(f"Precision@5:  {precision_5:.4f}")
    print(f"Precision@10: {precision_10:.4f}")
    print(f"AP:           {ap:.4f}")
    print(f"NDCG@5:       {ndcg_5:.4f}")
    print(f"NDCG@10:      {ndcg_10:.4f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("MULTIMODAL VECTOR DATABASE - COMPLETE DEMO")
    print("=" * 60)

    try:
        # Run demos
        demo_text_search()
        demo_image_search()
        demo_cross_modal_search()
        demo_index_comparison()
        demo_batch_processing()
        demo_evaluation_metrics()

        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
