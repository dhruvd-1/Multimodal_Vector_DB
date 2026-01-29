# Audio Retrieval Testing - Quick Start

## 📋 What Was Created

I've created comprehensive audio retrieval testing tools for your multimodal vector database:

1. **`demo_audio_retrieval.py`** - Interactive demo (recommended to start)
2. **`test_audio_quick.py`** - Quick automated test
3. **`test_audio_retrieval.py`** - Comprehensive test suite
4. **`AUDIO_RETRIEVAL_TESTING_GUIDE.md`** - Detailed documentation

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Test Audio Files

Create a folder with some audio files:

```bash
mkdir test_data
# Copy some .wav, .mp3, or .flac files into test_data/
```

Example structure:
```
test_data/
├── dog_bark.wav
├── music.mp3
└── speech.wav
```

### Step 2: Install Dependencies

Make sure you have CLAP installed:

```bash
pip install laion-clap
```

### Step 3: Run the Demo

```bash
python demo_audio_retrieval.py
```

This will:
- ✅ Find your audio files
- ✅ Initialize the search engine
- ✅ Index all audio files
- ✅ Let you search by audio (find similar audio)
- ✅ Let you search by text (e.g., "dog barking")

## 📊 What The Tests Do

### Audio-to-Audio Search
Search for similar audio files. For example:
- Input: `dog_bark.wav`
- Output: Other dog barking sounds ranked by similarity

**Expected Result**: The query audio itself should be rank #1 with distance ≈ 0

### Text-to-Audio Search
Search audio using text descriptions. For example:
- Input: `"piano music"`
- Output: Piano audio files ranked by relevance

**Expected Result**: Audio matching the description should rank highest

## 🎯 Verification

The test verifies:
1. ✅ Audio files can be indexed
2. ✅ Embeddings are generated (512D vectors)
3. ✅ Query audio returns itself as top result
4. ✅ Similar audio ranks higher than different audio
5. ✅ Text queries retrieve relevant audio

## 📚 Available Scripts

### 1. Interactive Demo (Recommended)
```bash
python demo_audio_retrieval.py
```
**Best for**: First-time testing, understanding how it works

### 2. Quick Test
```bash
python test_audio_quick.py
python test_audio_quick.py path/to/audio/folder  # Custom folder
```
**Best for**: Quick verification after changes

### 3. Comprehensive Test
```bash
python test_audio_retrieval.py
```
**Best for**: Detailed testing with custom test cases

## 🔍 Understanding Results

### Distance Metrics
- `< 0.01`: Essentially identical (query audio itself)
- `0.01 - 0.5`: Very similar
- `0.5 - 1.0`: Somewhat similar
- `> 1.0`: Different

### Similarity Score
Calculated as: `100 / (1 + distance)`
- Higher = More similar
- Range: 0% to 100%

## 💡 Example Usage in Code

```python
from src.retrieval import MultiModalSearchEngine

# Initialize
engine = MultiModalSearchEngine(embedding_dim=512, device='cpu')
engine.initialize(modalities=['audio'])

# Index audio
engine.ingest_content(
    'path/to/audio.wav',
    content_type='audio',
    metadata={'description': 'dog barking'}
)

# Search by audio
results = engine.search(
    query='path/to/query.wav',
    query_type='audio',
    k=5
)

# Search by text
text_embedding = engine.audio_embedder.embed_text("dog barking")
results = engine.vector_index.search(text_embedding, k=5)
```

## 🐛 Troubleshooting

### "No audio files found"
- Create `test_data/` folder
- Add .wav, .mp3, or .flac files
- Or specify path: `python test_audio_quick.py your/folder`

### "ImportError: laion_clap"
```bash
pip install laion-clap
```

### "Model not loaded"
The script should handle this, but ensure:
```python
embedder.load_model()  # Must be called before use
```

### Out of memory
For mobile/low memory:
```python
embedder = AudioEmbedder(max_batch_size=1, memory_limit_mb=512)
```

## 📖 Full Documentation

See **`AUDIO_RETRIEVAL_TESTING_GUIDE.md`** for:
- Detailed architecture explanation
- Advanced usage examples
- Performance optimization tips
- Troubleshooting guide
- API reference

## ✅ Success Criteria

Your audio retrieval is working correctly if:

1. ✅ Audio files load without errors
2. ✅ Query audio appears as rank #1 result (distance < 0.01)
3. ✅ Similar audio ranks higher than dissimilar audio
4. ✅ Text query "dog" returns dog-related audio
5. ✅ Results are consistent across runs

## 🎓 How It Works

### CLAP Model
- Uses Contrastive Language-Audio Pretraining
- Trained on 600K+ audio-text pairs
- Produces 512D embeddings
- Audio and text in same semantic space

### Search Process
1. Audio → 512D embedding vector
2. Store in vector index (FAISS)
3. Query → embedding vector
4. Find nearest neighbors (L2 distance)
5. Return ranked results

### Cross-Modal Magic
Text and audio embeddings are in the same space, so:
- Text "dog barking" ≈ Dog audio embedding
- Enables text → audio search
- No training needed!

## 🚀 Next Steps

1. **Run the demo**: `python demo_audio_retrieval.py`
2. **Test with your audio**: Add your files to `test_data/`
3. **Try text queries**: "music", "speech", "animal sounds"
4. **Integrate into your app**: Use the code examples
5. **Read the guide**: Check `AUDIO_RETRIEVAL_TESTING_GUIDE.md`

## 📞 Questions?

- Check error messages in output
- Review `AUDIO_RETRIEVAL_TESTING_GUIDE.md`
- Verify audio files are valid
- Test with simple cases first
- Check file paths are correct

---

**Happy Testing! 🎵**
