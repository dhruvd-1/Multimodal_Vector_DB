"""
Test Image-Text Retrieval Correctness on Flickr30k

Metrics:
- Image-to-Text Retrieval: Given an image, retrieve its captions (Recall@1, @5, @10)
- Text-to-Image Retrieval: Given a caption, retrieve the correct image (Recall@1, @5, @10)
"""
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedders.image_embedder import ImageEmbedder
from src.embedders.text_embedder import TextEmbedder

# Paths
IMG_DIR = r"data\raw\archive\flickr30k_images"
CAPTIONS_FILE = r"data\raw\archive\captions.txt"


def recall_at_k(similarities, ground_truth_idx, k_values=[1, 5, 10]):
    """
    Calculate Recall@K metrics.

    Args:
        similarities: Array of similarity scores [num_queries, num_targets]
        ground_truth_idx: List of correct indices for each query
        k_values: List of K values to compute recall at

    Returns:
        Dictionary with Recall@K scores
    """
    recalls = {}

    for k in k_values:
        correct = 0
        for i, gt_idx in enumerate(ground_truth_idx):
            # Get top-k predictions
            top_k = np.argsort(similarities[i])[::-1][:k]
            if gt_idx in top_k:
                correct += 1

        recalls[f"R@{k}"] = (correct / len(ground_truth_idx)) * 100

    return recalls


def test_text_to_image_retrieval(num_samples=500):
    """
    Text-to-Image Retrieval: Given a caption, find the correct image.
    """
    print("\n" + "=" * 70)
    print("TEXT-TO-IMAGE RETRIEVAL")
    print("=" * 70)

    # Load data
    print("\n1. Loading dataset...")
    df = pd.read_csv(CAPTIONS_FILE)

    # Sample unique images
    unique_images = df['image_name'].unique()[:num_samples]
    print(f"✓ Testing on {len(unique_images)} images")

    # Create test set: one caption per image
    test_data = []
    for img in unique_images:
        captions = df[df['image_name'] == img]['comment'].tolist()
        test_data.append({
            'image': img,
            'caption': captions[0],  # Use first caption for each image
        })

    # Initialize embedders
    print("\n2. Loading models...")
    img_embedder = ImageEmbedder(device="cpu")
    txt_embedder = TextEmbedder(device="cpu")
    img_embedder.load_model()
    txt_embedder.load_model()

    # Embed all images
    print(f"\n3. Embedding {len(unique_images)} images...")
    image_paths = [os.path.join(IMG_DIR, img) for img in unique_images]
    image_embeddings = img_embedder.batch_embed(image_paths, batch_size=16)
    print(f"✓ Image embeddings: {image_embeddings.shape}")

    # Embed all captions
    print(f"\n4. Embedding {len(test_data)} captions...")
    captions = [item['caption'] for item in test_data]
    caption_embeddings = txt_embedder.batch_embed(captions, batch_size=32)
    print(f"✓ Caption embeddings: {caption_embeddings.shape}")

    # Compute similarities
    print("\n5. Computing similarities...")
    # Shape: [num_captions, num_images]
    similarities = np.dot(caption_embeddings, image_embeddings.T)

    # Ground truth: caption i should match image i
    ground_truth = list(range(len(test_data)))

    # Calculate metrics
    print("\n6. Computing Recall@K metrics...")
    recalls = recall_at_k(similarities, ground_truth)

    print("\n" + "=" * 70)
    print("RESULTS: Text-to-Image Retrieval")
    print("=" * 70)
    for metric, value in recalls.items():
        print(f"{metric}: {value:.2f}%")

    # Show some examples
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS:")
    print("=" * 70)
    for i in range(min(3, len(test_data))):
        top_5 = np.argsort(similarities[i])[::-1][:5]
        correct_rank = np.where(np.argsort(similarities[i])[::-1] == i)[0][0] + 1

        print(f"\nQuery {i+1}: \"{test_data[i]['caption']}\"")
        print(f"Ground Truth: {test_data[i]['image']} (Rank: {correct_rank})")
        print(f"Top-5 Predictions:")
        for rank, idx in enumerate(top_5, 1):
            marker = "✓" if idx == i else "✗"
            print(f"  {marker} {rank}. {unique_images[idx]} (sim: {similarities[i][idx]:.4f})")

    # Cleanup
    img_embedder.unload_model()
    txt_embedder.unload_model()

    return recalls


def test_image_to_text_retrieval(num_samples=500):
    """
    Image-to-Text Retrieval: Given an image, find its captions.
    Note: Each image has 5 captions, so we check if any of them are in top-K.
    """
    print("\n" + "=" * 70)
    print("IMAGE-TO-TEXT RETRIEVAL")
    print("=" * 70)

    # Load data
    print("\n1. Loading dataset...")
    df = pd.read_csv(CAPTIONS_FILE)

    # Sample unique images
    unique_images = df['image_name'].unique()[:num_samples]
    print(f"✓ Testing on {len(unique_images)} images")

    # Create caption pool (all captions)
    caption_pool = []
    image_to_caption_idx = {}  # Maps image idx to list of caption indices

    for img_idx, img in enumerate(unique_images):
        captions = df[df['image_name'] == img]['comment'].tolist()
        image_to_caption_idx[img_idx] = []
        for cap in captions:
            image_to_caption_idx[img_idx].append(len(caption_pool))
            caption_pool.append(cap)

    print(f"✓ Total captions in pool: {len(caption_pool)}")

    # Initialize embedders
    print("\n2. Loading models...")
    img_embedder = ImageEmbedder(device="cpu")
    txt_embedder = TextEmbedder(device="cpu")
    img_embedder.load_model()
    txt_embedder.load_model()

    # Embed all images
    print(f"\n3. Embedding {len(unique_images)} images...")
    image_paths = [os.path.join(IMG_DIR, img) for img in unique_images]
    image_embeddings = img_embedder.batch_embed(image_paths, batch_size=16)

    # Embed all captions
    print(f"\n4. Embedding {len(caption_pool)} captions...")
    caption_embeddings = txt_embedder.batch_embed(caption_pool, batch_size=32)

    # Compute similarities
    print("\n5. Computing similarities...")
    # Shape: [num_images, num_captions]
    similarities = np.dot(image_embeddings, caption_embeddings.T)

    # Calculate metrics (at least one correct caption in top-K)
    print("\n6. Computing Recall@K metrics...")
    recalls = {}
    k_values = [1, 5, 10]

    for k in k_values:
        correct = 0
        for img_idx in range(len(unique_images)):
            top_k = np.argsort(similarities[img_idx])[::-1][:k]
            # Check if any ground truth caption is in top-k
            gt_caption_indices = image_to_caption_idx[img_idx]
            if any(cap_idx in top_k for cap_idx in gt_caption_indices):
                correct += 1

        recalls[f"R@{k}"] = (correct / len(unique_images)) * 100

    print("\n" + "=" * 70)
    print("RESULTS: Image-to-Text Retrieval")
    print("=" * 70)
    for metric, value in recalls.items():
        print(f"{metric}: {value:.2f}%")

    # Cleanup
    img_embedder.unload_model()
    txt_embedder.unload_model()

    return recalls


def main():
    """Run complete evaluation on Flickr30k."""
    print("=" * 70)
    print("FLICKR30K IMAGE-TEXT RETRIEVAL EVALUATION")
    print("=" * 70)

    num_samples = 500  # Test on 500 images (adjust based on your needs)

    # Test Text-to-Image
    t2i_results = test_text_to_image_retrieval(num_samples)

    # Test Image-to-Text
    i2t_results = test_image_to_text_retrieval(num_samples)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Dataset: Flickr30k ({num_samples} images)")
    print(f"\nText-to-Image:")
    for k, v in t2i_results.items():
        print(f"  {k}: {v:.2f}%")
    print(f"\nImage-to-Text:")
    for k, v in i2t_results.items():
        print(f"  {k}: {v:.2f}%")

    print("\n" + "=" * 70)
    print("✓ EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
