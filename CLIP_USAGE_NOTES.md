# CLIP Model Usage Notes

## Issue: Single Word Embeddings vs. Descriptive Phrases

### Problem
When testing semantic similarity with single words, CLIP may produce unexpected results:

```python
# ❌ Problematic: Single words
"cat" vs "kitten": 0.8886
"cat" vs "car":    0.8916  # car is MORE similar than kitten (incorrect!)
```

### Root Cause
CLIP models (Contrastive Language-Image Pre-training) are trained on image-caption pairs, where captions are typically full sentences or descriptive phrases like:
- "a photo of a cat"
- "a dog playing in the park"
- "a red car on the street"

The model learns to align **descriptive text** with images, not isolated words. Single words lack the contextual structure the model expects.

### Solution
Always use descriptive phrases when working with CLIP text embeddings:

```python
# ✅ Correct: Descriptive phrases
"a photo of a cat" vs "a photo of a kitten": 0.9294
"a photo of a cat" vs "a photo of a car":    0.7583  # Correct ordering!
```

### Best Practices

1. **For Image-Text Retrieval**: Use phrases like "a photo of {object}"
   ```python
   embedder.embed("a photo of a cat")
   ```

2. **For Semantic Search**: Use natural sentences
   ```python
   embedder.embed("a person walking a dog in the park")
   ```

3. **Avoid**: Single isolated words unless you have a specific reason
   ```python
   # Avoid this:
   embedder.embed("cat")
   ```

### When Single Words Might Work
- When comparing words within the same category (e.g., "dog" vs "puppy" vs "house")
- When the difference in semantic distance is large (e.g., "cat" vs "dog" vs "car")
- For very common objects that appear frequently in image captions

### Testing Implications
When writing tests for CLIP-based embedders:
- Use descriptive phrases that match CLIP's training distribution
- Don't expect perfect semantic alignment with single words
- Test with realistic use cases (image captions, search queries, etc.)

### Related Files
- [test_text_embedder.py](scripts/test_text_embedder.py) - Updated to use descriptive phrases
- [text_embedder.py](src/embedders/text_embedder.py) - CLIP-based text embedder implementation
