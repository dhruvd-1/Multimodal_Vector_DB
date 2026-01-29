"""Test caption-to-image retrieval on Flickr30k"""
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedders.image_embedder import ImageEmbedder
from src.embedders.text_embedder import TextEmbedder

# Paths
IMG_DIR = r"data\raw\archive\flickr30k_images"
CAPTIONS_FILE = r"data\raw\archive\captions.txt"

def caption_to_image_retrieval(query_caption, top_k=5, num_images=100):
    """
    Given a text caption, retrieve the most similar images.

    Args:
        query_caption: Text query (e.g., "a dog playing in the park")
        top_k: Number of top results to return
        num_images: Number of images to search through (set lower for faster testing)
    """
    print("=" * 60)
    print("Caption-to-Image Retrieval")
    print("=" * 60)

    # Load captions
    print("\n1. Loading dataset...")
    df = pd.read_csv(CAPTIONS_FILE)
    unique_images = df['image_name'].unique()[:num_images]
    print(f"✓ Using {len(unique_images)} images")

    # Initialize embedders
    print("\n2. Initializing embedders...")
    img_embedder = ImageEmbedder(device="cpu")
    txt_embedder = TextEmbedder(device="cpu")
    img_embedder.load_model()
    txt_embedder.load_model()

    # Embed query caption
    print("\n3. Embedding query caption...")
    print(f'Query: "{query_caption}"')
    query_embedding = txt_embedder.embed(query_caption)
    print(f"✓ Query embedding shape: {query_embedding.shape}")

    # Embed all images
    print(f"\n4. Embedding {len(unique_images)} images...")
    image_paths = [os.path.join(IMG_DIR, img) for img in unique_images]
    image_embeddings = img_embedder.batch_embed(image_paths, batch_size=8)
    print(f"✓ Image embeddings shape: {image_embeddings.shape}")

    # Compute similarities
    print("\n5. Computing similarities...")
    similarities = np.dot(image_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Display results
    print(f"\n6. Top {top_k} results:")
    print("=" * 60)
    for rank, idx in enumerate(top_indices, 1):
        img_name = unique_images[idx]
        similarity = similarities[idx]
        img_captions = df[df['image_name'] == img_name]['comment'].tolist()

        print(f"\nRank {rank}: {img_name} (similarity: {similarity:.4f})")
        print(f"  Path: {os.path.join(IMG_DIR, img_name)}")
        print(f"  Captions:")
        for i, cap in enumerate(img_captions[:3], 1):  # Show first 3 captions
            print(f"    {i}. {cap}")

    # Cleanup
    img_embedder.unload_model()
    txt_embedder.unload_model()

    print("\n" + "=" * 60)
    print("✓ RETRIEVAL COMPLETE!")
    print("=" * 60)

    return [(unique_images[idx], similarities[idx]) for idx in top_indices]


if __name__ == "__main__":
    # Example queries - try your own!
    queries = [
        "a dog running in the park",
        "people playing soccer",
        "a child eating ice cream",
    ]

    # Test with first query
    caption_to_image_retrieval(queries[0], top_k=5, num_images=100)

    # Uncomment to test more queries:
    # for query in queries:
    #     print("\n\n")
    #     caption_to_image_retrieval(query, top_k=3, num_images=50)
