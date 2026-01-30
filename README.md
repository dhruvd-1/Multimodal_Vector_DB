# Multimodal Vector Database

A high-performance cross-modal search system supporting text, images, video, and audio with unified HNSW indexing powered by CLIP and CLAP embeddings.

## Features

- **Cross-Modal Search**: Query with text, retrieve from images, videos, audio, and text documents
- **Dual Embedding Spaces**: CLIP for visual/text, CLAP for audio/text
- **Unified HNSW Index**: Single graph structure for all 44,444+ vectors
- **Individual Modality Indices**: Optimized per-modality search (6.74x faster)
- **Matryoshka Embeddings**: Adaptive dimensionality (64D, 128D, 256D, 512D)
- **Efficient Search**: ~13ms cross-modal, ~2ms individual modality

## System Overview

### Dataset
- **Images**: 31,783 from Flickr30k
- **Videos**: 7,010 from TrainValVideo
- **Audio**: 2,000 from ESC-50
- **Text**: 3,651 Wikipedia articles

**Total**: 44,444 indexed items

### Embedding Models

**CLIP (Images, Videos, Text)**
- Model: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
- Dimension: 512D
- Space: S_CLIP ⊂ R^512

**CLAP (Audio, Text)**
- Model: LAION CLAP (music_speech with fusion)
- Dimension: 512D
- Space: S_CLAP ⊂ R^512

Note: S_CLIP ≠ S_CLAP (incompatible embedding spaces)

## Architecture

```
src/
├── embedders/
│   ├── text_embedder.py      # CLIP text encoder
│   ├── image_embedder.py     # CLIP image encoder
│   ├── video_embedder.py     # CLIP video encoder (frame-based)
│   ├── audio_embedder.py     # CLAP audio encoder
│   └── projection.py         # Matryoshka projection layers
├── database/
│   └── vector_index.py       # HNSW index wrapper (hnswlib)
└── utils/
    └── metrics.py
```

## Installation

```bash
git clone <repository-url>
cd multimodal-vector-db

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Build Indices

**Build unified cross-modal index:**
```bash
python build_cross_modal_index.py
```

**Build individual modality indices:**
```bash
python build_all_indices.py
```

**Build Matryoshka indices (optional):**
```bash
python build_matryoshka_indices.py
```

### 2. Search Across Modalities

```python
from search_cross_modal import compare_modalities

# Get top 5 results from each modality
compare_modalities("dog playing in nature", k_per_modality=5)
```

### 3. Search Individual Modality

```bash
# Search images
python search_images.py

# Search videos
python search_videos.py

# Search audio
python search_audio.py

# Search text
python search_text.py
```

## Cross-Modal Search Architecture

### Dual-Encoder Strategy

Given query q:
1. Encode with CLIP: q_CLIP = f_CLIP(q)
2. Encode with CLAP: q_CLAP = f_CLAP(q)
3. Search HNSW with q_CLIP → get images, videos, text
4. Search HNSW with q_CLAP → get audio
5. Filter and combine results

### Why Two Encoders?

CLIP and CLAP are trained separately:
- **CLIP space**: Semantic similarity for images/videos/text
- **CLAP space**: Semantic similarity for audio/text
- Similarities are NOT comparable across spaces
- Each encoder queries the same unified HNSW index

### HNSW Index Structure

```
Unified Index: 44,444 vectors in R^512
├── Rows 0-31782:    Images (CLIP)
├── Rows 31783-38792: Videos (CLIP)
├── Rows 38793-40792: Audio (CLAP)
└── Rows 40793-44443: Text (CLIP)
```

Parameters:
- M = 20 (bidirectional links per node)
- ef_construction = 128
- metric = cosine similarity

## Benchmark Results

### Performance Summary

| Metric | Unified Index | Individual Indices |
|--------|---------------|-------------------|
| Average latency | 13.15ms | 1.95ms |
| Speedup | 1.0x | 6.74x |
| CLIP→Audio contamination | 0.0/100 | N/A |
| CLAP→Other contamination | 20.0/100 | N/A |

### Top Similarity Scores

| Query | Image | Video | Audio | Text |
|-------|-------|-------|-------|------|
| dog playing | 0.33 | 0.30 | 0.42 | 0.71 |
| music & instruments | 0.28 | 0.26 | 0.32 | 0.85 |
| space & astronomy | 0.27 | 0.34 | 0.36 | 0.81 |

**Run full benchmark:**
```bash
python benchmark_cross_modal.py
```

## Usage Examples

### Cross-Modal Search

```python
from search_cross_modal import search_cross_modal

# Search all modalities
results = search_cross_modal("dog playing in nature", k=20)

# Results are grouped by modality:
# results['image'] = list of image results
# results['video'] = list of video results
# results['audio'] = list of audio results
# results['text'] = list of text results
```

### Image Search

```python
from src.embedders.text_embedder import TextEmbedder
from src.database.vector_index import VectorIndex

# Load index
index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
index.load("saved_indices/image_index")

# Load embedder
embedder = TextEmbedder(device="cpu")
embedder.load_model()

# Search
query_emb = embedder.embed("sunset over mountains")
results = index.search(query_emb, k=10)

for result in results:
    print(f"Image: {result['metadata']['image_name']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Caption: {result['metadata']['captions'][0]}")
```

### Audio Search with CLAP

```python
from src.embedders.audio_embedder import AudioEmbedder
from src.database.vector_index import VectorIndex

# Load audio index
index = VectorIndex(dimension=512, index_type='hnsw', metric='cosine')
index.load("saved_indices/audio_index")

# Load CLAP embedder
embedder = AudioEmbedder(model_name='music_speech', device='cpu', enable_fusion=True)
embedder.load_model()

# Search with text query
query_emb = embedder.embed_text("dog barking")
results = index.search(query_emb, k=10)

for result in results:
    print(f"Audio: {result['metadata']['filename']}")
    print(f"Category: {result['metadata']['category']}")
    print(f"Similarity: {result['similarity']:.4f}")
```

## Matryoshka Embeddings

Nested dimensionality for adaptive precision:

```
v_512 = [v_64 | v_128 | v_256 | v_512]
```

| Dimension | Storage | Speed | Accuracy |
|-----------|---------|-------|----------|
| 64D | 8x smaller | Fastest | 85-90% |
| 128D | 4x smaller | Very fast | 92-95% |
| 256D | 2x smaller | Fast | 96-98% |
| 512D | Full size | Baseline | 100% |

**Convert weights:**
```bash
python convert_matryoshka_weights.py
```

**Build Matryoshka indices:**
```bash
python build_matryoshka_indices.py
```

## Index Selection Guide

| Scenario | Recommended |
|----------|-------------|
| Multi-modal exploration | Unified index |
| Known target modality | Individual indices |
| Speed-critical | Individual indices |
| Memory-constrained | Unified index |
| Prototyping | Unified index |
| Production | Individual indices |

## Configuration Files

### Data Paths

Edit paths in build scripts:
```python
IMG_DIR = r"data\raw\archive\flickr30k_images"
CAPTIONS_FILE = r"data\raw\archive\captions.txt"
VIDEO_DIR = r"data\raw\archive (2)\TrainValVideo"
AUDIO_DIR = r"data\raw\archive (1)\audio\audio"
AUDIO_LABELS = r"data\raw\archive (1)\esc50.csv"
WIKI_DIR1 = r"data\raw\archive(3)\1of2"
WIKI_DIR2 = r"data\raw\archive(3)\2of2"
```

### HNSW Parameters

Tune in `VectorIndex.build_index()`:
```python
vector_index.build_index(
    max_elements=50000,     # Max vectors
    ef_construction=128,    # Build quality (higher = better, slower)
    M=20                    # Links per node (higher = better recall)
)
vector_index.index.set_ef(50)  # Search quality
```

## Performance Tips

1. **GPU Acceleration**: Set `device='cuda'` for 5-10x faster embedding
2. **Batch Processing**: Use `batch_embed()` for multiple items
3. **Index Selection**: HNSW for speed, individual indices for precision
4. **Matryoshka**: Use 128D for 4x speedup with minimal quality loss
5. **CLAP Fusion**: Enable `enable_fusion=True` for better audio quality

## System Requirements

- Python 3.8+
- 16GB+ RAM (for full dataset)
- GPU optional (NVIDIA CUDA for acceleration)
- Storage: ~5GB for datasets, ~500MB for indices

## Documentation

- **System Architecture**: See `cross_modal_architecture.tex` for detailed documentation
- **Matryoshka Guide**: See `MATRYOSHKA_WEIGHTS_ISSUE.md` for projection layer details
- **Benchmark Report**: Run `python benchmark_cross_modal.py` for full analysis

## Future Improvements

### 1. Unified Embedding Space
Replace CLIP + CLAP with:
- **ImageBind** (Meta): 6 modalities in one space
- **LanguageBind**: Extended CLIP with audio
- **OneLLM**: Unified multimodal LLM

Benefits:
- Single encoder for all modalities
- Comparable cross-modal similarities
- True cross-modal ranking
- 2x faster search

### 2. Hybrid Retrieval
Combine dense + sparse:
```
score(q, d) = α·dense(q, d) + (1-α)·sparse(q, d)
```
- Dense: CLIP/CLAP (semantic)
- Sparse: BM25 (lexical)

### 3. Advanced Features
- [ ] Temporal modeling for videos
- [ ] Query expansion with LLMs
- [ ] Re-ranking with cross-encoders
- [ ] Duplicate detection
- [ ] Distributed indices (multi-node)

## Troubleshooting

**CLAP not finding audio files:**
- Ensure `enable_fusion=True` matches index build settings
- Check CLAP model variant (music_speech vs. general)

**Out of memory:**
- Use Matryoshka 128D indices
- Reduce batch size
- Build indices sequentially (one modality at a time)

**Low similarity scores:**
- Normalize embeddings (done automatically)
- Check embedding space (CLIP vs. CLAP)
- Verify data quality

## Citation

```bibtex
@software{multimodal_cross_modal_db,
  title={Cross-Modal Multimodal Search System},
  author={},
  year={2025},
  note={CLIP + CLAP unified HNSW index}
}
```

## Acknowledgments

- CLIP by OpenAI
- CLAP by LAION
- HNSWlib by Malkov & Yashunin
- Flickr30k, ESC-50, Wikipedia datasets

## License

MIT License
