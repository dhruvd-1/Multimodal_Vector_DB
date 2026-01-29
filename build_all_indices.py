"""
Build and save HNSW indices for ALL datasets (Image, Video, Audio)
Run this once to index everything, then use search scripts to query
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embedders.image_embedder import ImageEmbedder
from src.embedders.video_embedder import VideoEmbedder
from src.embedders.audio_embedder import AudioEmbedder
from src.database.vector_index import VectorIndex

# Data paths
IMG_DIR = r"data\raw\archive\flickr30k_images"
CAPTIONS_FILE = r"data\raw\archive\captions.txt"
VIDEO_DIR = r"data\raw\archive (2)\TrainValVideo"
AUDIO_DIR = r"data\raw\archive (1)\audio\audio"
AUDIO_LABELS = r"data\raw\archive (1)\esc50.csv"

# Index save paths
INDICES_DIR = "saved_indices"
os.makedirs(INDICES_DIR, exist_ok=True)


def build_image_index():
    """Build HNSW index for ALL Flickr30k images"""
    print("\n" + "="*80)
    print("BUILDING IMAGE INDEX")
    print("="*80)

    # Load captions
    print("\n1. Loading Flickr30k dataset...")
    df = pd.read_csv(CAPTIONS_FILE)
    unique_images = df['image_name'].unique()
    print(f"✓ Found {len(unique_images)} images")

    # Initialize embedder
    print("\n2. Initializing image embedder...")
    img_embedder = ImageEmbedder(device="cpu")
    img_embedder.load_model()

    # Build index
    print("\n3. Building HNSW index...")
    vector_index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
    vector_index.build_index(max_elements=len(unique_images) + 100)

    # Prepare metadata
    print("\n4. Preparing metadata...")
    image_metadata = []
    for img_name in unique_images:
        captions = df[df['image_name'] == img_name]['comment'].tolist()
        image_metadata.append({
            'image_name': img_name,
            'image_path': os.path.join(IMG_DIR, img_name),
            'captions': captions,
            'modality': 'image'
        })

    # Embed all images
    print(f"\n5. Embedding {len(unique_images)} images...")
    image_paths = [os.path.join(IMG_DIR, img) for img in unique_images]
    batch_size = 16
    all_embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_embeddings = img_embedder.batch_embed(batch, batch_size=batch_size)
        all_embeddings.append(batch_embeddings)
        print(f"  Progress: {min(i + batch_size, len(image_paths))}/{len(image_paths)}", end='\r')

    all_embeddings = np.vstack(all_embeddings)
    print(f"\n✓ Generated embeddings: {all_embeddings.shape}")

    # Add to index
    print("\n6. Adding to HNSW index...")
    vector_index.add_vectors(all_embeddings, image_metadata)

    # Save index
    save_path = os.path.join(INDICES_DIR, "image_index")
    print(f"\n7. Saving index to {save_path}...")
    vector_index.save(save_path)

    img_embedder.unload_model()

    print(f"\n✓ IMAGE INDEX COMPLETE!")
    print(f"   Indexed: {len(unique_images)} images")
    print(f"   Saved to: {save_path}")
    return save_path


def build_video_index():
    """Build HNSW index for ALL videos"""
    print("\n" + "="*80)
    print("BUILDING VIDEO INDEX")
    print("="*80)

    # Find all videos
    print("\n1. Finding video files...")
    if not os.path.exists(VIDEO_DIR):
        print(f"⚠️  Video directory not found: {VIDEO_DIR}")
        print("   Skipping video indexing...")
        return None

    video_files = [f for f in os.listdir(VIDEO_DIR)
                   if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if len(video_files) == 0:
        print("⚠️  No video files found, skipping...")
        return None

    print(f"✓ Found {len(video_files)} videos")

    # Initialize embedder
    print("\n2. Initializing video embedder...")
    vid_embedder = VideoEmbedder(device="cpu", sample_fps=1.0, max_frames=30)
    vid_embedder.load_model()

    # Build index
    print("\n3. Building HNSW index...")
    vector_index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
    vector_index.build_index(max_elements=len(video_files) + 100)

    # Prepare metadata
    print("\n4. Preparing metadata...")
    video_metadata = []
    for vid_file in video_files:
        video_metadata.append({
            'video_name': vid_file,
            'video_path': os.path.join(VIDEO_DIR, vid_file),
            'modality': 'video'
        })

    # Embed all videos
    print(f"\n5. Embedding {len(video_files)} videos...")
    video_paths = [os.path.join(VIDEO_DIR, vid) for vid in video_files]
    batch_size = 4
    all_embeddings = []

    for i in range(0, len(video_paths), batch_size):
        batch = video_paths[i:i + batch_size]
        batch_embeddings = vid_embedder.batch_embed(batch, batch_size=batch_size)
        all_embeddings.append(batch_embeddings)
        print(f"  Progress: {min(i + batch_size, len(video_paths))}/{len(video_paths)}", end='\r')

    all_embeddings = np.vstack(all_embeddings)
    print(f"\n✓ Generated embeddings: {all_embeddings.shape}")

    # Add to index
    print("\n6. Adding to HNSW index...")
    vector_index.add_vectors(all_embeddings, video_metadata)

    # Save index
    save_path = os.path.join(INDICES_DIR, "video_index")
    print(f"\n7. Saving index to {save_path}...")
    vector_index.save(save_path)

    vid_embedder.unload_model()

    print(f"\n✓ VIDEO INDEX COMPLETE!")
    print(f"   Indexed: {len(video_files)} videos")
    print(f"   Saved to: {save_path}")
    return save_path


def build_audio_index():
    """Build HNSW index for ALL audio files (ESC-50)"""
    print("\n" + "="*80)
    print("BUILDING AUDIO INDEX (ESC-50)")
    print("="*80)

    # Load audio labels
    print("\n1. Loading ESC-50 dataset...")
    df = pd.read_csv(AUDIO_LABELS)
    print(f"✓ Found {len(df)} audio files")
    print(f"✓ Categories: {df['category'].nunique()} unique sound categories")

    # Initialize embedder
    print("\n2. Initializing audio embedder...")
    audio_embedder = AudioEmbedder(model_name='music_speech', device='cpu', enable_fusion=True)
    audio_embedder.load_model()

    # Build index
    print("\n3. Building HNSW index...")
    vector_index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
    vector_index.build_index(max_elements=len(df) + 100)

    # Prepare metadata
    print("\n4. Preparing metadata...")
    audio_metadata = []
    for _, row in df.iterrows():
        audio_metadata.append({
            'filename': row['filename'],
            'audio_path': os.path.join(AUDIO_DIR, row['filename']),
            'category': row['category'],
            'fold': row['fold'],
            'esc10': row['esc10'],
            'modality': 'audio'
        })

    # Embed all audio
    print(f"\n5. Embedding {len(df)} audio files...")
    audio_paths = [os.path.join(AUDIO_DIR, fname) for fname in df['filename']]
    batch_size = 8
    all_embeddings = []

    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i + batch_size]
        batch_embeddings = audio_embedder.batch_embed(batch, batch_size=batch_size)
        all_embeddings.append(batch_embeddings)
        print(f"  Progress: {min(i + batch_size, len(audio_paths))}/{len(audio_paths)}", end='\r')

    all_embeddings = np.vstack(all_embeddings)
    print(f"\n✓ Generated embeddings: {all_embeddings.shape}")

    # Add to index
    print("\n6. Adding to HNSW index...")
    vector_index.add_vectors(all_embeddings, audio_metadata)

    # Save index
    save_path = os.path.join(INDICES_DIR, "audio_index")
    print(f"\n7. Saving index to {save_path}...")
    vector_index.save(save_path)

    audio_embedder.unload_model()

    print(f"\n✓ AUDIO INDEX COMPLETE!")
    print(f"   Indexed: {len(df)} audio files")
    print(f"   Saved to: {save_path}")
    return save_path


def main():
    """Build all indices"""
    print("\n" + "="*80)
    print("🚀 BUILDING ALL HNSW INDICES")
    print("="*80)
    print("\nThis will index all your datasets:")
    print("  - Images: Flickr30k (~31,000 images)")
    print("  - Videos: TrainValVideo")
    print("  - Audio: ESC-50 (~2,000 audio files)")
    print("  - Text: Wikipedia Simple English (~249k articles)")
    print("\nIndices will be saved to:", os.path.abspath(INDICES_DIR))
    print("\n" + "="*80)

    results = {}

    # Build image index
    try:
        results['image'] = build_image_index()
    except Exception as e:
        print(f"\n❌ Image indexing failed: {e}")
        import traceback
        traceback.print_exc()

    # Build video index
    try:
        results['video'] = build_video_index()
    except Exception as e:
        print(f"\n❌ Video indexing failed: {e}")
        import traceback
        traceback.print_exc()

    # Build audio index
    try:
        results['audio'] = build_audio_index()
    except Exception as e:
        print(f"\n❌ Audio indexing failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("✅ ALL INDICES BUILT!")
    print("="*80)
    for modality, path in results.items():
        if path:
            print(f"  {modality.upper()}: {path}")
        else:
            print(f"  {modality.upper()}: SKIPPED")

    print(f"\nIndices saved in: {os.path.abspath(INDICES_DIR)}")
    print("\n⚠️  NOTE: Text index must be built separately (takes longer)")
    print("   Run: python build_text_index.py")
    print("\nNext steps:")
    print("  - Use search_images.py to search images")
    print("  - Use search_videos.py to search videos")
    print("  - Use search_audio.py to search audio")
    print("  - Use search_text.py to search Wikipedia articles (after building text index)")


if __name__ == "__main__":
    main()
