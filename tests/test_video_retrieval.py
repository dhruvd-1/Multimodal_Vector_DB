import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedders.video_embedder import VideoEmbedder
from src.embedders.text_embedder import TextEmbedder

def video_caption_retrieval(video_dir, query_caption, top_k=5):
    """
    Given a text caption, retrieve the most similar videos.

    Args:
        video_dir: Directory containing video files
        query_caption: Text query (e.g., "a person dancing")
        top_k: Number of top results to return
    """
   
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    


    vid_embedder = VideoEmbedder(device="cpu", sample_fps=1.0, max_frames=30)
    txt_embedder = TextEmbedder(device="cpu")
    vid_embedder.load_model()
    txt_embedder.load_model()

    query_embedding = txt_embedder.embed(query_caption)

    # Embed all videos
    print(f"\n4. Embedding {len(video_files)} videos...")
    video_paths = [os.path.join(video_dir, vid) for vid in video_files]
    video_embeddings = []

    for i, vid_path in enumerate(video_paths[:31]):
        print(f"  Processing {i+1}/{len(video_files)}: {video_files[i]}")
        try:
            embedding = vid_embedder.embed(vid_path)
            video_embeddings.append(embedding)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            video_embeddings.append(np.zeros(512))  # Fallback

    video_embeddings = np.array(video_embeddings)
    print(f"✓ Video embeddings shape: {video_embeddings.shape}")

    # Compute similarities
    print("\n5. Computing similarities...")
    similarities = np.dot(video_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Display results
    print(f"\n6. Top {top_k} results:")
    print("=" * 60)
    for rank, idx in enumerate(top_indices, 1):
        vid_name = video_files[idx]
        similarity = similarities[idx]

        print(f"\nRank {rank}: {vid_name}")
        print(f"  Path: {os.path.join(video_dir, vid_name)}")
        print(f"  Similarity: {similarity:.4f}")

    # Cleanup
    vid_embedder.unload_model()
    txt_embedder.unload_model()

    print("\n" + "=" * 60)
    print("✓ RETRIEVAL COMPLETE!")
    print("=" * 60)

    return [(video_files[idx], similarities[idx]) for idx in top_indices]


def main():
    """Example usage"""
    
    # Set your video directory
    video_dir = r"data/raw/archive (2)/TrainValVideo"  

    queries = [
        "math equations on a blackboard",
        "girl speaking and walking",
        "guys sitting in a car",
        "factory",
    ]
    video_caption_retrieval(video_dir, queries[:2], top_k=5)


if __name__ == "__main__":
    main()
