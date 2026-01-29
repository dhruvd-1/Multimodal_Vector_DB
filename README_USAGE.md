# Multimodal Database - Quick Start Guide

## 🚀 Setup (One-Time)

### Step 1: Build HNSW Indices for ALL Datasets

This indexes all your data (images, videos, audio, text) and saves the indices to disk.
**You only need to run this ONCE** (or when you add new data).

```bash
# Build image, video, audio indices
python build_all_indices.py

# Build text index separately (takes longer ~10-20 min)
python build_text_index.py
```

This will:
- ✅ Index ~31,000 Flickr30k images
- ✅ Index all videos in TrainValVideo
- ✅ Index ~2,000 ESC-50 audio files
- ✅ Index ~249,000 Wikipedia articles
- ✅ Save indices to `saved_indices/` folder

**Time:** ~15-30 minutes total depending on your system

---

## 🔍 Searching (Fast & Easy)

Once indices are built, you can search instantly!

### Search Images
```bash
python search_images.py
```

Or search with your own query:
```bash
python -c "from search_images import search_images; search_images('a cat playing with yarn')"
```

### Search Videos
```bash
python search_videos.py
```

Or search with your own query:
```bash
python -c "from search_videos import search_videos; search_videos('someone dancing')"
```

### Search Audio
```bash
python search_audio.py
```

Or search with your own query:
```bash
python -c "from search_audio import search_audio; search_audio('dog barking')"
```

### Search Text (Wikipedia)
```bash
python search_text.py
```

Or search with your own query:
```bash
python -c "from search_text import search_text; search_text('quantum physics')"
```

---

## 📂 File Structure

```
Multimodal db/
├── build_all_indices.py        # Build image/video/audio indices
├── build_text_index.py          # Build text index (separate, larger)
├── search_images.py             # Search images with text
├── search_videos.py             # Search videos with text
├── search_audio.py              # Search audio with text
├── search_text.py               # Search Wikipedia articles
├── saved_indices/               # Saved HNSW indices (persistent)
│   ├── image_index.*            # Image HNSW index + metadata
│   ├── video_index.*            # Video HNSW index + metadata
│   ├── audio_index.*            # Audio HNSW index + metadata
│   └── text_index.*             # Text HNSW index + metadata
├── data/raw/                    # Your datasets
│   ├── archive/                 # Flickr30k images
│   ├── archive (1)/             # ESC-50 audio
│   ├── archive (2)/             # Videos
│   └── archive(3)/              # Wikipedia Simple English text
└── src/                         # Source code
    ├── embedders/               # Image, video, audio, text embedders
    └── database/                # Vector index (HNSW)
```

---

## 🎯 Key Features

✅ **Fast HNSW Search**: O(log n) instead of O(n) brute force
✅ **Persistent Indices**: Build once, search forever
✅ **FP16 Compression**: 50% smaller index size
✅ **Cross-Modal Search**: Text → Image/Video/Audio
✅ **Batch Processing**: Efficient embedding generation

---

## 💡 Tips

1. **Indices are saved to disk** - you don't need to rebuild them unless you add new data
2. **Search is instant** - HNSW index loads in <1 second
3. **Modify queries** - edit the search scripts or use command line
4. **Add new data** - just re-run `build_all_indices.py`

---

## 🛠️ Troubleshooting

**Index not found?**
```bash
python build_all_indices.py
```

**Slow searching?**
- Indices should load instantly (<1 sec)
- If slow, check if `saved_indices/` folder exists

**Want to rebuild indices?**
```bash
rm -rf saved_indices/
python build_all_indices.py
```

---

## 📊 Dataset Info

- **Images**: Flickr30k (~31,000 images with captions)
- **Videos**: TrainValVideo
- **Audio**: ESC-50 (~2,000 environmental sounds, 50 categories)
- **Text**: Wikipedia Simple English (~249,000 articles, 31M tokens)

---

## 🚀 Next Steps

1. Build indices: `python build_all_indices.py`
2. Build text index: `python build_text_index.py`
3. Search images: `python search_images.py`
4. Search videos: `python search_videos.py`
5. Search audio: `python search_audio.py`
6. Search text: `python search_text.py`
7. Customize queries in each script!
