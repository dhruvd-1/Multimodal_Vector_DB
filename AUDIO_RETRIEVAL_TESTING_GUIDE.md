# Audio Retrieval Testing Guide

This guide explains how to test audio retrieval functionality in your multimodal vector database.

## Overview

The system supports two types of audio retrieval:
1. **Audio-to-Audio**: Find similar audio files (e.g., find songs similar to a given song)
2. **Text-to-Audio**: Search audio using text descriptions (e.g., "dog barking" → retrieve dog barking audio)

## Quick Start

### 1. Prepare Test Data

Create a folder with audio files:

```
test_data/
├── dog_bark.wav
├── cat_meow.wav
├── piano_music.mp3
└── ocean_waves.wav
```

Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`

### 2. Run Quick Test

```bash
python test_audio_quick.py
```

Or specify a custom folder:

```bash
python test_audio_quick.py path/to/your/audio/folder
```

### 3. Run Comprehensive Test

```bash
python test_audio_retrieval.py
```

## Test Scripts

### `test_audio_quick.py` - Quick Testing

**Purpose**: Fast verification that audio retrieval works

**Features**:
- Automatically finds audio files in a folder
- Indexes all audio files
- Searches for similar audio
- Verifies top result is the query itself
- Shows text-to-audio example

**Use when**: You just want to verify the system works

### `test_audio_retrieval.py` - Comprehensive Testing

**Purpose**: Detailed testing with custom audio files and metadata

**Features**:
- Define specific test cases with descriptions
- Test audio-to-audio search
- Test text-to-audio search with multiple queries
- Verify results match expectations
- Detailed logging and statistics

**Use when**: You need thorough testing with specific test cases

## How Audio Retrieval Works

### 1. Audio Embedding

```python
from src.embedders import AudioEmbedder

embedder = AudioEmbedder(model_name='music_speech', device='cpu')
embedder.load_model()

# Generate embedding for audio file
embedding = embedder.embed('path/to/audio.wav')  # Returns 512D vector
```

### 2. Indexing Audio

```python
from src.retrieval import MultiModalSearchEngine

engine = MultiModalSearchEngine(embedding_dim=512, device='cpu')
engine.initialize(modalities=['audio'])

# Add audio to index
engine.ingest_content(
    'path/to/audio.wav',
    content_type='audio',
    metadata={'description': 'dog barking', 'category': 'animals'}
)
```

### 3. Audio-to-Audio Search

Find similar audio files:

```python
results = engine.search(
    query='path/to/query_audio.wav',
    query_type='audio',
    k=5,  # Return top 5 results
    filter_content_type='audio'
)

for result in results:
    print(f"File: {result['metadata']['description']}")
    print(f"Distance: {result['distance']}")
```

### 4. Text-to-Audio Search

Search audio using text description:

```python
# Get text embedding in audio space
text_embedding = engine.audio_embedder.embed_text("dog barking")

# Search
results = engine.vector_index.search(
    text_embedding,
    k=5,
    filter_fn=lambda m: m.get('content_type') == 'audio'
)
```

## Understanding Results

### Distance Metrics

- **Distance**: Lower = More similar
  - `< 0.01`: Essentially identical
  - `0.01 - 0.5`: Very similar
  - `0.5 - 1.0`: Similar
  - `> 1.0`: Different

- **Similarity Score**: `1 / (1 + distance)`
  - Higher = More similar
  - Range: 0 to 1

### Expected Behavior

#### Audio-to-Audio Search:
- ✅ Query audio should be rank #1 with distance ≈ 0
- ✅ Similar audio (same category) should rank higher
- ✅ Different audio should rank lower

#### Text-to-Audio Search:
- ✅ Audio matching text description should rank high
- ✅ Semantically similar audio should rank higher than unrelated
- ⚠️  Exact keyword matching not guaranteed (uses semantic understanding)

## Example Test Cases

### Test Case 1: Similar Audio Detection

```python
audio_files = [
    ("dog_bark_1.wav", "dog barking"),
    ("dog_bark_2.wav", "dog barking"),
    ("cat_meow.wav", "cat meowing"),
]

# Expected: Searching with dog_bark_1 should return dog_bark_2 as 2nd result
```

### Test Case 2: Cross-Modal Search

```python
# Index various animal sounds
audio_files = [
    ("dog.wav", "dog barking"),
    ("cat.wav", "cat meowing"),
    ("bird.wav", "bird chirping"),
]

# Query: "dog barking"
# Expected: dog.wav should rank highest
```

### Test Case 3: Music Classification

```python
audio_files = [
    ("piano.wav", "classical piano music"),
    ("guitar.wav", "rock guitar solo"),
    ("drums.wav", "drum beat"),
]

# Query: "piano music"
# Expected: piano.wav should rank highest
```

## Troubleshooting

### Issue: ImportError for laion_clap

**Solution**:
```bash
pip install laion-clap
```

### Issue: "Model not loaded" error

**Solution**: Call `load_model()` before using embedder:
```python
embedder.load_model()
```

### Issue: Out of memory on mobile

**Solution**: Use smaller batch sizes:
```python
embedder = AudioEmbedder(max_batch_size=1, memory_limit_mb=512)
```

### Issue: Audio file not found

**Solution**: Use absolute paths:
```python
import os
audio_path = os.path.abspath('path/to/audio.wav')
```

### Issue: Text-to-audio search returns poor results

**Possible causes**:
1. Text description too vague - be more specific
2. Audio doesn't match description semantically
3. Model not trained on similar content

**Solutions**:
- Use descriptive text ("dog barking loudly" vs "sound")
- Try different CLAP models: 'music_speech', 'music', 'general'
- Verify audio content matches description

## Advanced Usage

### Custom Audio Embedder

```python
embedder = AudioEmbedder(
    model_name='music',  # Specialized for music
    device='cuda',  # Use GPU
    enable_fusion=True,  # Better quality, slower
    max_batch_size=8
)
```

### Save and Load Index

```python
# Save
engine.save('./saved_audio_db')

# Load later
engine = MultiModalSearchEngine(storage_path='./saved_audio_db')
engine.initialize(modalities=['audio'])
engine.load()
```

### Batch Processing

```python
# Efficient batch ingestion
audio_paths = ['audio1.wav', 'audio2.wav', 'audio3.wav']
metadata_list = [
    {'description': 'dog'},
    {'description': 'cat'},
    {'description': 'bird'}
]

engine.batch_ingest(
    contents=audio_paths,
    content_type='audio',
    metadata_list=metadata_list
)
```

## Performance Tips

1. **Use GPU if available**: `device='cuda'`
2. **Batch similar operations**: Use `batch_embed()` for multiple files
3. **Choose appropriate index**: `flat` for small datasets, `hnsw` for large
4. **Mobile optimization**: Set `max_batch_size` and `memory_limit_mb`
5. **Model selection**: Use specialized models when possible

## Verification Checklist

- [ ] Audio files load without errors
- [ ] Embeddings generated successfully (512D vectors)
- [ ] Query audio returns itself as top result (distance < 0.01)
- [ ] Similar audio ranks higher than dissimilar audio
- [ ] Text queries return semantically relevant audio
- [ ] Search completes in reasonable time
- [ ] Results are reproducible

## Next Steps

1. Test with your own audio dataset
2. Experiment with different text queries
3. Try different CLAP models for your use case
4. Integrate into your application
5. Optimize for your specific requirements

## Reference

- **CLAP Models**: Uses Contrastive Language-Audio Pretraining
- **Embedding Dimension**: 512D vectors
- **Similarity Metric**: L2 distance (Euclidean)
- **Index Types**: flat (exact), hnsw (approximate)

## Questions?

If you encounter issues:
1. Check error messages carefully
2. Verify audio files are valid
3. Ensure all dependencies installed
4. Test with simple cases first
5. Check file paths are correct
