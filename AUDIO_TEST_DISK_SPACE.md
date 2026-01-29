# Audio Embedder Test - Disk Space Issue

## Problem
The audio embedder test failed with:
```
OSError: [Errno 28] No space left on device
```

Your C: drive is currently **full (0 bytes free)**.

## Root Cause
The CLAP (Contrastive Language-Audio Pretraining) model needs to download:
- **~1.8GB** model checkpoint on first use
- Downloaded to: `%USERPROFILE%\.cache\laion_clap` (typically `C:\Users\<username>\.cache\laion_clap`)

## Solutions

### Option 1: Free Up Disk Space (Recommended)
Free up at least **2GB** on your C: drive:

1. **Clean temporary files**:
   ```powershell
   # Run Disk Cleanup
   cleanmgr /d C:
   ```

2. **Clean Python cache**:
   ```powershell
   # Remove pip cache
   pip cache purge
   
   # Remove Python __pycache__ directories
   Get-ChildItem -Path C:\Users -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
   ```

3. **Clean Hugging Face cache** (if you have it):
   ```powershell
   # Remove old models from Hugging Face cache
   Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\*"
   ```

4. **Move large folders** to another drive:
   - Downloads folder
   - OneDrive cache
   - Other project folders

### Option 2: Change CLAP Cache Location
Set CLAP to use a different drive with more space:

```python
# In your code, before loading the model:
import os
os.environ['LAION_CLAP_CACHE'] = 'D:\\.cache\\laion_clap'  # Use D: drive
```

Or set environment variable in PowerShell:
```powershell
$env:LAION_CLAP_CACHE = "D:\.cache\laion_clap"
python scripts/test_audio_embedder.py
```

### Option 3: Use Smaller Mobile Model
CLAP offers a smaller "HTSAT-tiny" model for mobile devices:

```python
embedder = AudioEmbedder(device='cpu')
embedder._is_mobile = True  # Force mobile mode
embedder.load_model()
```

This uses a smaller model but may have slightly lower accuracy.

### Option 4: Skip Audio Tests Temporarily
If you don't need audio embeddings right now, you can skip this test:

```bash
# Just test text embeddings
python scripts/test_text_embedder.py
```

## Current System Status
- **C: Drive Used**: 209.89 GB
- **C: Drive Free**: 0 bytes ❌
- **Required Space**: ~2 GB for CLAP model

## Testing After Freeing Space

Once you have freed up disk space, run:

```bash
python scripts/test_audio_embedder.py
```

The model will download once, then future runs will use the cached version.

## Model Information

**CLAP Model Details**:
- **Purpose**: Audio-Text cross-modal embeddings
- **Size**: ~1.8 GB (larger_clap_music_and_speech)
- **Embedding Dimension**: 512D
- **Use Case**: Search audio files using text queries
- **Training**: Pre-trained on 600K+ audio-text pairs

**Alternative Models** (if space is limited):
- `music`: Music-only model (~1.5 GB)
- `general`: General audio model (~1.7 GB)
- Mobile: HTSAT-tiny variant (smaller)

## Note on Text Prompts

Like CLIP, CLAP works better with **descriptive phrases** rather than single words:

✅ Good: "a dog barking loudly", "piano music playing"
❌ Avoid: "dog", "piano"

This is reflected in the updated test file.
