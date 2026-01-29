# Multimodal Database - Quick Start Guide

## 🚀 Setup (One-Time)

### Step 1: Build HNSW Indices for ALL Datasets

This indexes all your data (images, videos, audio) and saves the indices to disk.
**You only need to run this ONCE** (or when you add new data).

```bash
python build_all_indices.py
```

This will:
- ✅ Index ~31,000 Flickr30k images
- ✅ Index all videos in TrainValVideo
- ✅ Index ~2,000 ESC-50 audio files
- ✅ Save indices to `saved_indices/` folder

**Time:** ~10-20 minutes depending on your system

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

---

## 📂 File Structure

```
Multimodal db/
├── build_all_indices.py        # Build & save HNSW indices (run once)
├── search_images.py             # Search images with text
├── search_videos.py             # Search videos with text
├── search_audio.py              # Search audio with text
├── saved_indices/               # Saved HNSW indices (persistent)
│   ├── image_index.index        # Image HNSW index
│   ├── image_index.metadata     # Image metadata
│   ├── video_index.index        # Video HNSW index
│   ├── video_index.metadata     # Video metadata
│   ├── audio_index.index        # Audio HNSW index
│   └── audio_index.metadata     # Audio metadata
├── data/raw/                    # Your datasets
│   ├── archive/                 # Flickr30k images
│   ├── archive (1)/             # ESC-50 audio
│   └── archive (2)/             # Videos
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

---

## 🚀 Next Steps

1. Build indices: `python build_all_indices.py`
2. Search images: `python search_images.py`
3. Search videos: `python search_videos.py`
4. Search audio: `python search_audio.py`
5. Customize queries in each script!
