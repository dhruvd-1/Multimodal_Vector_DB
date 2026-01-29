"""Test cross-modal retrieval across images and videos"""
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedders.image_embedder import ImageEmbedder
from src.embedders.video_embedder import VideoEmbedder
from src.embedders.text_embedder import TextEmbedder


def cross_modal_retrieval(image_dir, video_dir, query_caption, top_k=10):
    

    # Get files
    print("\n1. Loading media files...")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:20]
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))][:20]

    print(f"✓ Found {len(image_files)} images (limited to 20)")
    print(f"✓ Found {len(video_files)} videos (limited to 20)")

    # Initialize embedders
    print("\n2. Initializing embedders...")
    img_embedder = ImageEmbedder(device="cpu")
    vid_embedder = VideoEmbedder(device="cpu", sample_fps=1.0, max_frames=30)
    txt_embedder = TextEmbedder(device="cpu")

    img_embedder.load_model()
    vid_embedder.load_model()
    txt_embedder.load_model()

    # Embed query
    print("\n3. Embedding query caption...")
    print(f'Query: "{query_caption}"')
    query_embedding = txt_embedder.embed(query_caption)

    # Store all embeddings with metadata
    all_embeddings = []
    all_metadata = []

    # Embed images
    print(f"\n4. Embedding {len(image_files)} images...")
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        print(f"  Processing image {i+1}/{len(image_files)}: {img_file}")
        try:
            embedding = img_embedder.embed(img_path)
            all_embeddings.append(embedding)
            all_metadata.append({
                'type': 'image',
                'filename': img_file,
                'path': img_path
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_embeddings.append(np.zeros(512))
            all_metadata.append({
                'type': 'image',
                'filename': img_file,
                'path': img_path,
                'error': str(e)
            })

    # Embed videos
    print(f"\n5. Embedding {len(video_files)} videos...")
    for i, vid_file in enumerate(video_files):
        vid_path = os.path.join(video_dir, vid_file)
        print(f"  Processing video {i+1}/{len(video_files)}: {vid_file}")
        try:
            embedding = vid_embedder.embed(vid_path)
            all_embeddings.append(embedding)
            all_metadata.append({
                'type': 'video',
                'filename': vid_file,
                'path': vid_path
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_embeddings.append(np.zeros(512))
            all_metadata.append({
                'type': 'video',
                'filename': vid_file,
                'path': vid_path,
                'error': str(e)
            })

    # Convert to array
    all_embeddings = np.array(all_embeddings)
    print(f"\n✓ Total embeddings: {all_embeddings.shape}")

    # Compute similarities
    print("\n6. Computing similarities across all media...")
    similarities = np.dot(all_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Display results
    print(f"\n7. Top {top_k} results:")
    print("=" * 60)

    results = []
    for rank, idx in enumerate(top_indices, 1):
        metadata = all_metadata[idx]
        similarity = similarities[idx]

        media_type = metadata['type'].upper()
        filename = metadata['filename']
        path = metadata['path']

        print(f"\nRank {rank}: [{media_type}] {filename}")
        print(f"  Path: {path}")
        print(f"  Similarity: {similarity:.4f}")

        if 'error' in metadata:
            print(f"  ⚠ Had error: {metadata['error']}")

        results.append((metadata['type'], path, similarity))

    # Cleanup
    img_embedder.unload_model()
    vid_embedder.unload_model()
    txt_embedder.unload_model()

    print("\n" + "=" * 60)
    print("✓ CROSS-MODAL RETRIEVAL COMPLETE!")
    print("=" * 60)

    return results


def main():
    """Example usage"""
    # Set directories
    image_dir = r"data/raw/archive/flickr30k_images"
    video_dir = r"data/raw/archive (2)/TrainValVideo"

    # Check directories exist
    if not os.path.exists(image_dir):
        print(f"✗ Image directory not found: {image_dir}")
        return

    if not os.path.exists(video_dir):
        print(f"✗ Video directory not found: {video_dir}")
        return

    # Example queries
    queries = [
        "a dog playing in the park",
        "people dancing",
        "a person riding a bicycle",
        "a cat",
    ]

    # Run cross-modal retrieval
    results = cross_modal_retrieval(image_dir, video_dir, queries[0], top_k=10)

    # Summary
    print("\nSummary:")
    image_count = sum(1 for r in results if r[0] == 'image')
    video_count = sum(1 for r in results if r[0] == 'video')
    print(f"  Images in top 10: {image_count}")
    print(f"  Videos in top 10: {video_count}")


if __name__ == "__main__":
    main()
