"""
Build unified cross-modal HNSW index
Combines all modalities (image, video, audio, text) into a single searchable index
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
from src.embedders.text_embedder import TextEmbedder
from src.database.vector_index import VectorIndex

IMG_DIR = r"data\raw\archive\flickr30k_images"
CAPTIONS_FILE = r"data\raw\archive\captions.txt"
VIDEO_DIR = r"data\raw\archive (2)\TrainValVideo"
AUDIO_DIR = r"data\raw\archive (1)\audio\audio"
AUDIO_LABELS = r"data\raw\archive (1)\esc50.csv"
WIKI_DIR1 = r"data\raw\archive(3)\1of2"
WIKI_DIR2 = r"data\raw\archive(3)\2of2"

INDICES_DIR = "saved_indices"
os.makedirs(INDICES_DIR, exist_ok=True)


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


def build_cross_modal_index():
    """
    Build unified cross-modal HNSW index with ALL modalities

    Strategy:
    - Use CLIP embeddings (512D) for images and videos
    - Use CLAP embeddings (512D) for audio
    - Use CLIP text embeddings (512D) for text
    - All embeddings are 512D and can be searched together
    """
    print("\n" + "="*80)
    print("BUILDING CROSS-MODAL FUSION INDEX")
    print("="*80)
    print("\nCombining ALL modalities into single searchable index:")
    print("  - Images (Flickr30k)")
    print("  - Videos (TrainValVideo)")
    print("  - Audio (ESC-50)")
    print("  - Text (Wikipedia)")
    print("\nNote: CLIP and CLAP are in different embedding spaces.")
    print("      Images/Videos/Text use CLIP (aligned)")
    print("      Audio uses CLAP (separate space)")
    print("="*80)

    all_embeddings = []
    all_metadata = []
    total_items = 0

    print("\n" + "="*80)
    print("1. INDEXING IMAGES")
    print("="*80)

    df_img = pd.read_csv(CAPTIONS_FILE)
    unique_images = df_img['image_name'].unique()
    print(f"Found {len(unique_images)} images")

    print("Initializing image embedder...")
    img_embedder = ImageEmbedder(device="cpu")
    img_embedder.load_model()

    print(f"Embedding {len(unique_images)} images...")
    image_paths = [os.path.join(IMG_DIR, img) for img in unique_images]

    batch_size = 16
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_embeddings = img_embedder.batch_embed(batch, batch_size=batch_size)
        all_embeddings.append(batch_embeddings)

        for j, img_name in enumerate(unique_images[i:i + batch_size]):
            captions = df_img[df_img['image_name'] == img_name]['comment'].tolist()
            all_metadata.append({
                'modality': 'image',
                'image_name': img_name,
                'image_path': os.path.join(IMG_DIR, img_name),
                'captions': captions,
                'display_name': img_name,
                'preview': f"Image: {captions[0][:100]}..." if captions else img_name
            })

        print(f"  Progress: {min(i + batch_size, len(image_paths))}/{len(image_paths)}", end='\r')

    total_items += len(unique_images)
    print(f"\n Indexed {len(unique_images)} images")
    img_embedder.unload_model()

    print("\n" + "="*80)
    print("2. INDEXING VIDEOS")
    print("="*80)

    if os.path.exists(VIDEO_DIR):
        video_files = [f for f in os.listdir(VIDEO_DIR)
                      if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        if len(video_files) > 0:
            print(f"Found {len(video_files)} videos")

            print("Initializing video embedder...")
            vid_embedder = VideoEmbedder(device="cpu", sample_fps=1.0, max_frames=30)
            vid_embedder.load_model()

            print(f"Embedding {len(video_files)} videos...")
            video_paths = [os.path.join(VIDEO_DIR, vid) for vid in video_files]

            batch_size = 4
            for i in range(0, len(video_paths), batch_size):
                batch = video_paths[i:i + batch_size]
                batch_embeddings = vid_embedder.batch_embed(batch, batch_size=batch_size)
                all_embeddings.append(batch_embeddings)

                for j, vid_file in enumerate(video_files[i:i + batch_size]):
                    all_metadata.append({
                        'modality': 'video',
                        'video_name': vid_file,
                        'video_path': os.path.join(VIDEO_DIR, vid_file),
                        'display_name': vid_file,
                        'preview': f"Video: {vid_file}"
                    })

                print(f"  Progress: {min(i + batch_size, len(video_paths))}/{len(video_paths)}", end='\r')

            total_items += len(video_files)
            print(f"\n Indexed {len(video_files)} videos")
            vid_embedder.unload_model()
        else:
            print("  No video files found, skipping...")
    else:
        print(f"  Video directory not found, skipping...")

    print("\n" + "="*80)
    print("3. INDEXING AUDIO")
    print("="*80)

    df_audio = pd.read_csv(AUDIO_LABELS)
    print(f"Found {len(df_audio)} audio files")

    print("Initializing audio embedder...")
    audio_embedder = AudioEmbedder(model_name='music_speech', device='cpu', enable_fusion=True)
    audio_embedder.load_model()

    print(f"Embedding {len(df_audio)} audio files...")
    audio_paths = [os.path.join(AUDIO_DIR, fname) for fname in df_audio['filename']]

    batch_size = 8
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i + batch_size]
        batch_embeddings = audio_embedder.batch_embed(batch, batch_size=batch_size)
        all_embeddings.append(batch_embeddings)

        for j, (_, row) in enumerate(df_audio.iloc[i:i + batch_size].iterrows()):
            all_metadata.append({
                'modality': 'audio',
                'filename': row['filename'],
                'audio_path': os.path.join(AUDIO_DIR, row['filename']),
                'category': row['category'],
                'display_name': f"{row['category']} - {row['filename']}",
                'preview': f"Audio: {row['category']}"
            })

        print(f"  Progress: {min(i + batch_size, len(audio_paths))}/{len(audio_paths)}", end='\r')

    total_items += len(df_audio)
    print(f"\n Indexed {len(df_audio)} audio files")
    audio_embedder.unload_model()

    print("\n" + "="*80)
    print("4. INDEXING TEXT (Wikipedia - First 10,000 articles)")
    print("="*80)

    print("Finding Wikipedia files...")
    wiki_files = []
    for wiki_dir in [WIKI_DIR1, WIKI_DIR2]:
        if os.path.exists(wiki_dir):
            files = [os.path.join(wiki_dir, f) for f in os.listdir(wiki_dir)
                    if f.startswith('wiki_')]
            wiki_files.extend(files[:5])  # Just first 5 files for cross-modal

    print(f"Parsing articles from {len(wiki_files)} wiki files...")
    all_articles = []
    for wiki_file in wiki_files:
        articles = parse_wiki_file(wiki_file)
        all_articles.extend(articles)
        if len(all_articles) >= 10000:
            break

    all_articles = all_articles[:10000]  # Limit to 10k for cross-modal
    print(f"Using {len(all_articles)} Wikipedia articles")

    print("Initializing text embedder...")
    text_embedder = TextEmbedder(device="cpu")
    text_embedder.load_model()

    print(f"Embedding {len(all_articles)} articles...")
    texts_to_embed = [f"{a['title']}. {a['content'][:500]}" for a in all_articles]

    batch_size = 32
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i + batch_size]
        batch_embeddings = text_embedder.batch_embed(batch, batch_size=batch_size)
        all_embeddings.append(batch_embeddings)

        for j, article in enumerate(all_articles[i:i + batch_size]):
            all_metadata.append({
                'modality': 'text',
                'title': article['title'],
                'content': article['content'],
                'display_name': article['title'],
                'preview': article['preview']
            })

        print(f"  Progress: {min(i + batch_size, len(texts_to_embed))}/{len(texts_to_embed)}", end='\r')

    total_items += len(all_articles)
    print(f"\n Indexed {len(all_articles)} articles")
    text_embedder.unload_model()

    print("\n" + "="*80)
    print("5. BUILDING UNIFIED CROSS-MODAL INDEX")
    print("="*80)

    print(f"Total items: {total_items:,}")
    print("  Images:", len([m for m in all_metadata if m['modality'] == 'image']))
    print("  Videos:", len([m for m in all_metadata if m['modality'] == 'video']))
    print("  Audio:", len([m for m in all_metadata if m['modality'] == 'audio']))
    print("  Text:", len([m for m in all_metadata if m['modality'] == 'text']))

    print("\nStacking embeddings...")
    all_embeddings_matrix = np.vstack(all_embeddings)
    print(f" Combined embeddings: {all_embeddings_matrix.shape}")

    print("\nBuilding HNSW index...")
    vector_index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
    vector_index.build_index(max_elements=total_items + 1000)
    vector_index.add_vectors(all_embeddings_matrix, all_metadata)

    save_path = os.path.join(INDICES_DIR, "cross_modal_index")
    print(f"\nSaving index to {save_path}...")
    vector_index.save(save_path)

    print(f"\n CROSS-MODAL INDEX COMPLETE!")
    print(f"   Total items: {total_items:,}")
    print(f"   Saved to: {save_path}")

    return save_path


def main():
    """Build cross-modal index"""
    print("\n" + "="*80)
    print(" BUILDING CROSS-MODAL FUSION INDEX")
    print("="*80)
    print("\nThis creates a unified index combining:")
    print("  - Images + Videos + Text (CLIP space)")
    print("  - Audio (CLAP space)")
    print("\nNote: Different embedding spaces may affect cross-modal relevance")
    print("="*80)

    try:
        save_path = build_cross_modal_index()

        print("\n" + "="*80)
        print(" CROSS-MODAL INDEX BUILT!")
        print("="*80)
        print(f"  Saved to: {save_path}")
        print("\nNext step:")
        print("  - Use search_cross_modal.py to search across all modalities")

    except Exception as e:
        print(f"\n Cross-modal indexing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
