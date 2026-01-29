"""Test image embedder on Flickr30k dataset"""
import os
import pandas as pd
import numpy as np
from src.embedders.image_embedder import ImageEmbedder

# Paths
IMG_DIR = r"data\raw\archive\flickr30k_images"
CAPTIONS_FILE = r"data\raw\archive\captions.txt"

def test_flickr30k():
    print("=" * 60)
    print("Testing Image Embedder on Flickr30k")
    print("=" * 60)

    # Load captions
    print("\n1. Loading captions...")
    df = pd.read_csv(CAPTIONS_FILE)
    print(f"✓ Loaded {len(df)} captions for {df['image_name'].nunique()} images")

    # Initialize embedder
    print("\n2. Initializing embedder...")
    embedder = ImageEmbedder(device="cpu")  # Change to "cuda" if you have GPU
    embedder.load_model()

    # Test on a few images
    print("\n3. Testing on sample images...")
    sample_images = df['image_name'].unique()[:5]

    for img_name in sample_images:
        img_path = os.path.join(IMG_DIR, img_name)

        # Get captions for this image
        captions = df[df['image_name'] == img_name]['comment'].tolist()

        # Generate embedding
        embedding = embedder.embed(img_path)

        print(f"\n✓ {img_name}")
        print(f"  Shape: {embedding.shape}")
        print(f"  Norm: {np.linalg.norm(embedding):.6f}")
        print(f"  Captions:")
        for i, cap in enumerate(captions, 1):
            print(f"    {i}. {cap}")

    # Test batch embedding
    print("\n4. Testing batch embedding...")
    batch_paths = [os.path.join(IMG_DIR, img) for img in sample_images]
    batch_embeddings = embedder.batch_embed(batch_paths, batch_size=2)
    print(f"✓ Batch shape: {batch_embeddings.shape}")
    print(f"✓ All normalized: {np.allclose(np.linalg.norm(batch_embeddings, axis=1), 1.0)}")

    # Test similarity
    print("\n5. Testing similarity between images...")
    sim = np.dot(batch_embeddings[0], batch_embeddings[1])
    print(f"✓ Similarity between image 1 and 2: {sim:.4f}")

    embedder.unload_model()

    print("\n" + "=" * 60)
    print("✓ TEST COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    test_flickr30k()
