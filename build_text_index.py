"""
Build and save HNSW index for Wikipedia Simple English text documents
Parses wiki files and indexes individual articles
"""
import os
import sys
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embedders.text_embedder import TextEmbedder
from src.database.vector_index import VectorIndex
import numpy as np

# Data paths
WIKI_DIR1 = r"data\raw\archive(3)\1of2"
WIKI_DIR2 = r"data\raw\archive(3)\2of2"

# Index save path
INDICES_DIR = "saved_indices"
os.makedirs(INDICES_DIR, exist_ok=True)


def parse_wiki_file(file_path):
    """
    Parse a wiki file into individual articles

    Returns:
        list of dicts: [{'title': str, 'content': str}, ...]
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split by double newlines (articles are separated)
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

            # Skip very short articles or those without proper title
            if len(article_content) > 100 and len(title) > 0 and len(title) < 100:
                articles.append({
                    'title': title,
                    'content': article_content,
                    'preview': article_content[:200] + '...' if len(article_content) > 200 else article_content
                })

    return articles


def build_text_index():
    """Build HNSW index for ALL Wikipedia articles"""
    print("\n" + "="*80)
    print("BUILDING TEXT INDEX (Wikipedia Simple English)")
    print("="*80)

    # Find all wiki files
    print("\n1. Finding Wikipedia text files...")
    wiki_files = []
    for wiki_dir in [WIKI_DIR1, WIKI_DIR2]:
        if os.path.exists(wiki_dir):
            files = [os.path.join(wiki_dir, f) for f in os.listdir(wiki_dir)
                    if f.startswith('wiki_')]
            wiki_files.extend(files)

    print(f"✓ Found {len(wiki_files)} wiki files")

    # Parse all articles
    print("\n2. Parsing articles from wiki files...")
    all_articles = []
    for wiki_file in wiki_files:
        articles = parse_wiki_file(wiki_file)
        all_articles.extend(articles)
        print(f"  Parsed {len(all_articles)} articles so far...", end='\r')

    print(f"\n✓ Parsed {len(all_articles)} total articles")

    # Initialize embedder
    print("\n3. Initializing text embedder...")
    text_embedder = TextEmbedder(device="cpu")
    text_embedder.load_model()

    # Build index
    print("\n4. Building HNSW index...")
    vector_index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
    vector_index.build_index(max_elements=len(all_articles) + 1000)

    # Prepare metadata with source file info
    print("\n5. Preparing metadata...")
    article_metadata = []
    for article in all_articles:
        article_metadata.append({
            'title': article['title'],
            'content': article['content'],
            'preview': article['preview'],
            'modality': 'text'
        })

    # Embed all articles (use title + beginning of content for better semantic matching)
    print(f"\n6. Embedding {len(all_articles)} articles...")
    print("   (Using title + content preview for semantic matching)")

    texts_to_embed = []
    for article in all_articles:
        # Combine title with content preview for better search
        text = f"{article['title']}. {article['content'][:500]}"
        texts_to_embed.append(text)

    batch_size = 32
    all_embeddings = []

    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i + batch_size]
        batch_embeddings = text_embedder.batch_embed(batch, batch_size=batch_size)
        all_embeddings.append(batch_embeddings)
        print(f"  Progress: {min(i + batch_size, len(texts_to_embed))}/{len(texts_to_embed)}", end='\r')

    all_embeddings = np.vstack(all_embeddings)
    print(f"\n✓ Generated embeddings: {all_embeddings.shape}")

    # Add to index
    print("\n7. Adding to HNSW index...")
    vector_index.add_vectors(all_embeddings, article_metadata)

    # Save index
    save_path = os.path.join(INDICES_DIR, "text_index")
    print(f"\n8. Saving index to {save_path}...")
    vector_index.save(save_path)

    text_embedder.unload_model()

    print(f"\n✓ TEXT INDEX COMPLETE!")
    print(f"   Indexed: {len(all_articles)} Wikipedia articles")
    print(f"   Saved to: {save_path}")
    print(f"\n   Dataset: Wikipedia Simple English")
    print(f"   Articles: {len(all_articles):,}")
    print(f"   Total tokens: ~31M")
    return save_path


def main():
    """Build text index"""
    print("\n" + "="*80)
    print("🚀 BUILDING WIKIPEDIA TEXT INDEX")
    print("="*80)
    print("\nThis will index Wikipedia Simple English articles:")
    print("  - ~249,396 articles")
    print("  - ~31M tokens")
    print("  - ~196,000 unique words")
    print("\nIndex will be saved to:", os.path.abspath(INDICES_DIR))
    print("\n" + "="*80)

    try:
        save_path = build_text_index()

        print("\n" + "="*80)
        print("✅ TEXT INDEX BUILT!")
        print("="*80)
        print(f"  Saved to: {save_path}")
        print("\nNext step:")
        print("  - Use search_text.py to search Wikipedia articles")

    except Exception as e:
        print(f"\n❌ Text indexing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
