"""
Build HNSW indices using Matryoshka embeddings at different dimensions
Creates separate indices for 512D, 256D, 128D, 64D for comparison
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embedders.text_embedder import TextEmbedder
from src.embedders.image_embedder import ImageEmbedder
from src.embedders.video_embedder import VideoEmbedder
from src.database.vector_index import VectorIndex

# Data paths (same as build_cross_modal_index.py)
IMG_DIR = r"data\raw\archive\flickr30k_images"
CAPTIONS_FILE = r"data\raw\archive\captions.txt"
VIDEO_DIR = r"data\raw\archive (2)\TrainValVideo"
WIKI_DIR1 = r"data\raw\archive(3)\1of2"
WIKI_DIR2 = r"data\raw\archive(3)\2of2"


def build_image_indices_matryoshka():
    """Build Matryoshka image indices at all dimensions"""
    print("\n" + "="*80)
    print("BUILDING IMAGE INDICES (Matryoshka)")
    print("="*80 + "\n")

    # Load data from captions CSV
    df_img = pd.read_csv(CAPTIONS_FILE)
    unique_images = df_img['image_name'].unique()
    print(f"Found {len(unique_images)} unique images")

    image_paths = [os.path.join(IMG_DIR, img) for img in unique_images]

    # Load embedders
    print("\nLoading image embedder...")
    img_embedder = ImageEmbedder(device="cpu")
    img_embedder.load_model()

    print("Loading Matryoshka text embedder...")
    txt_embedder = TextEmbedder(
        device="cpu",
        use_matryoshka=True,
        matryoshka_dims=[512, 256, 128, 64]
    )
    txt_embedder.load_model()
    txt_embedder.load_matryoshka_weights('models/matryoshka_weights.pt')

    # Build indices at each dimension
    for dim in [512, 256, 128, 64]:
        print(f"\n{'─'*80}")
        print(f"Building {dim}D index...")
        print(f"{'─'*80}")

        # Create index
        vector_index = VectorIndex(dimension=dim, index_type='hnsw', metric='cosine')
        vector_index.build_index(max_elements=len(image_paths) + 100, ef_construction=128, M=20)

        # Generate embeddings
        embeddings = []
        metadata = []

        batch_size = 32
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Embedding images ({dim}D)"):
            batch = image_paths[i:i+batch_size]
            batch_embs = img_embedder.batch_embed(batch, output_dim=dim)

            for img_path, emb in zip(batch, batch_embs):
                embeddings.append(emb)
                img_name = os.path.basename(img_path)
                captions = df_img[df_img['image_name'] == img_name]['comment'].tolist()
                metadata.append({
                    'path': img_path,
                    'id': Path(img_path).stem,
                    'modality': 'image',
                    'captions': captions
                })

        # Add to index
        embeddings_matrix = np.vstack(embeddings)
        vector_index.add_vectors(embeddings_matrix, metadata)

        # Save
        save_path = f"saved_indices/image_matryoshka_{dim}d"
        vector_index.save(save_path)
        print(f"✓ Saved to {save_path}")

    img_embedder.unload_model()
    txt_embedder.unload_model()


def build_video_indices_matryoshka():
    """Build Matryoshka video indices at all dimensions"""
    print("\n" + "="*80)
    print("BUILDING VIDEO INDICES (Matryoshka)")
    print("="*80 + "\n")

    # Load video files
    if not os.path.exists(VIDEO_DIR):
        print(f"⚠️  Video directory not found: {VIDEO_DIR}")
        return

    video_files = [f for f in os.listdir(VIDEO_DIR)
                  if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if len(video_files) == 0:
        print("⚠️  No video files found, skipping...")
        return

    unique_videos = [os.path.join(VIDEO_DIR, vid) for vid in video_files]
    print(f"Found {len(unique_videos)} unique videos")

    # Load embedders
    print("\nLoading video embedder...")
    vid_embedder = VideoEmbedder(device="cpu")
    vid_embedder.load_model()

    print("Loading Matryoshka text embedder...")
    txt_embedder = TextEmbedder(
        device="cpu",
        use_matryoshka=True,
        matryoshka_dims=[512, 256, 128, 64]
    )
    txt_embedder.load_model()
    txt_embedder.load_matryoshka_weights('models/matryoshka_weights.pt')

    # Build indices at each dimension
    for dim in [512, 256, 128, 64]:
        print(f"\n{'─'*80}")
        print(f"Building {dim}D index...")
        print(f"{'─'*80}")

        vector_index = VectorIndex(dimension=dim, index_type='hnsw', metric='cosine')
        vector_index.build_index(max_elements=len(unique_videos) + 100, ef_construction=128, M=20)

        embeddings = []
        metadata = []

        for video_path in tqdm(unique_videos, desc=f"Embedding videos ({dim}D)"):
            try:
                emb = vid_embedder.embed(video_path, output_dim=dim)
                embeddings.append(emb)
                metadata.append({
                    'path': video_path,
                    'id': Path(video_path).stem,
                    'modality': 'video'
                })
            except Exception as e:
                print(f"\nSkipping {video_path}: {e}")
                continue

        embeddings_matrix = np.vstack(embeddings)
        vector_index.add_vectors(embeddings_matrix, metadata)

        save_path = f"saved_indices/video_matryoshka_{dim}d"
        vector_index.save(save_path)
        print(f"✓ Saved to {save_path}")

    vid_embedder.unload_model()
    txt_embedder.unload_model()


def parse_wiki_file(file_path):
    """Parse a wiki file into individual articles"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    articles = []
    chunks = content.split('\n\n\n')

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        lines = chunk.split('\n', 1)
        if len(lines) >= 2:
            title = lines[0].strip()
            article_content = lines[1].strip()

            if len(article_content) > 100 and len(title) > 0 and len(title) < 100:
                articles.append({
                    'title': title,
                    'content': article_content,
                    'preview': article_content[:200] + '...' if len(article_content) > 200 else article_content
                })

    return articles


def build_text_indices_matryoshka():
    """Build Matryoshka text indices at all dimensions"""
    print("\n" + "="*80)
    print("BUILDING TEXT INDICES (Matryoshka)")
    print("="*80 + "\n")

    # Parse Wikipedia data from multiple files
    print("Finding Wikipedia files...")
    wiki_files = []
    for wiki_dir in [WIKI_DIR1, WIKI_DIR2]:
        if os.path.exists(wiki_dir):
            files = [os.path.join(wiki_dir, f) for f in os.listdir(wiki_dir)
                    if f.startswith('wiki_')]
            wiki_files.extend(files[:5])  # First 5 files

    if len(wiki_files) == 0:
        print(f"❌ No Wikipedia files found in {WIKI_DIR1} or {WIKI_DIR2}")
        return

    print(f"Parsing articles from {len(wiki_files)} wiki files...")
    all_articles = []
    for wiki_file in wiki_files:
        articles = parse_wiki_file(wiki_file)
        all_articles.extend(articles)
        if len(all_articles) >= 10000:
            break

    all_articles = all_articles[:10000]  # Limit to 10k
    print(f"Found {len(all_articles)} articles")

    # Load embedder
    print("\nLoading Matryoshka text embedder...")
    txt_embedder = TextEmbedder(
        device="cpu",
        use_matryoshka=True,
        matryoshka_dims=[512, 256, 128, 64]
    )
    txt_embedder.load_model()
    txt_embedder.load_matryoshka_weights('models/matryoshka_weights.pt')

    # Build indices at each dimension
    for dim in [512, 256, 128, 64]:
        print(f"\n{'─'*80}")
        print(f"Building {dim}D index...")
        print(f"{'─'*80}")

        vector_index = VectorIndex(dimension=dim, index_type='hnsw', metric='cosine')
        vector_index.build_index(max_elements=len(all_articles) + 100, ef_construction=128, M=20)

        embeddings = []
        metadata = []

        batch_size = 64
        for i in tqdm(range(0, len(all_articles), batch_size), desc=f"Embedding text ({dim}D)"):
            batch = all_articles[i:i+batch_size]
            texts = [f"{a['title']}. {a['content'][:500]}" for a in batch]

            batch_embs = txt_embedder.batch_embed(texts, output_dim=dim)

            for article, emb in zip(batch, batch_embs):
                embeddings.append(emb)
                metadata.append({
                    'title': article['title'],
                    'content': article['content'],
                    'preview': article['preview'],
                    'modality': 'text'
                })

        embeddings_matrix = np.vstack(embeddings)
        vector_index.add_vectors(embeddings_matrix, metadata)

        save_path = f"saved_indices/text_matryoshka_{dim}d"
        vector_index.save(save_path)
        print(f"✓ Saved to {save_path}")

    txt_embedder.unload_model()


def main():
    print("\n" + "="*80)
    print("MATRYOSHKA INDICES BUILDER")
    print("Building indices at 512D, 256D, 128D, 64D")
    print("="*80)

    # Create output directory
    Path("saved_indices").mkdir(exist_ok=True)

    # Build all indices
    try:
        build_image_indices_matryoshka()
    except Exception as e:
        print(f"\n❌ Error building image indices: {e}")
        import traceback
        traceback.print_exc()

    try:
        build_video_indices_matryoshka()
    except Exception as e:
        print(f"\n❌ Error building video indices: {e}")
        import traceback
        traceback.print_exc()

    try:
        build_text_indices_matryoshka()
    except Exception as e:
        print(f"\n❌ Error building text indices: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("✅ DONE! Built Matryoshka indices at all dimensions")
    print("="*80)
    print("\nNext step: Run benchmark_matryoshka_full.py to compare quality")


if __name__ == "__main__":
    main()
