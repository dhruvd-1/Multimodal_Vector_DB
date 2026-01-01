import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import os
import time
import psutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

!pip install -q open_clip_torch transformers

import open_clip
from transformers import AutoProcessor, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Paths
FLICKR_IMG_DIR = "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images"
FLICKR_CAPTIONS = "/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv"

NUM_SAMPLES = 1000
BATCH_SIZE = 32

# Load data
df = pd.read_csv(FLICKR_CAPTIONS, sep='|')
df.columns = [c.strip() for c in df.columns]
df_filtered = df[df['comment_number'].astype(str).str.strip() == '0'].head(NUM_SAMPLES).copy()

image_paths = [os.path.join(FLICKR_IMG_DIR, str(img).strip()) for img in df_filtered['image_name'].values]
captions = [str(c).strip() for c in df_filtered['comment'].values]

# Load images
images = []
for p in tqdm(image_paths, desc="Loading images"):
    try:
        images.append(Image.open(p).convert('RGB'))
    except:
        images.append(Image.new('RGB', (224, 224)))

print(f"✓ Loaded {len(images)} image-caption pairs\n")

# Model configs
MODELS = [
    {
        'name': 'OpenCLIP-ViT-B/32',
        'type': 'openclip',
        'model_name': 'ViT-B-32',
        'pretrained': 'openai'
    },
    {
        'name': 'OpenCLIP-ViT-S/32',
        'type': 'openclip',
        'model_name': 'ViT-S-32',
        'pretrained': 'openai'
    },
    {
        'name': 'SigLIP-Base',
        'type': 'siglip',
        'model_name': 'google/siglip-base-patch16-224'
    }
]

results = []

for config in MODELS:
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"{'='*70}")

    try:
        # Memory before
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024

        # Load model
        load_start = time.time()

        if config['type'] == 'openclip':
            model, _, preprocess = open_clip.create_model_and_transforms(
                config['model_name'],
                pretrained=config['pretrained']
            )
            model = model.to(DEVICE).eval()
            tokenizer = open_clip.get_tokenizer(config['model_name'])

        else:  # siglip
            model = AutoModel.from_pretrained(config['model_name']).to(DEVICE).eval()
            processor = AutoProcessor.from_pretrained(config['model_name'])

        load_time = time.time() - load_start
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = (total_params * 4) / (1024 * 1024)  # FP32

        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"  Parameters: {total_params:,} ({model_size_mb:.1f}MB)")
        print(f"  RAM: {mem_after - mem_before:.1f}MB")

        # Encode images
        img_times = []
        img_emb = []

        for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Encoding images"):
            batch_imgs = images[i:i+BATCH_SIZE]

            start = time.perf_counter()
            with torch.no_grad():
                if config['type'] == 'openclip':
                    batch = torch.stack([preprocess(img) for img in batch_imgs]).to(DEVICE)
                    emb = model.encode_image(batch)
                else:
                    inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
                    emb = model.get_image_features(**inputs)

                img_emb.append(emb.cpu())

            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            img_times.append(time.perf_counter() - start)

        img_emb = torch.cat(img_emb)
        avg_img_time = (sum(img_times) / len(img_times)) * 1000 / BATCH_SIZE

        # Encode texts
        txt_times = []
        txt_emb = []

        for i in tqdm(range(0, len(captions), BATCH_SIZE), desc="Encoding texts"):
            batch_txt = captions[i:i+BATCH_SIZE]

            start = time.perf_counter()
            with torch.no_grad():
                if config['type'] == 'openclip':
                    batch = tokenizer(batch_txt).to(DEVICE)
                    emb = model.encode_text(batch)
                else:
                    inputs = processor(text=batch_txt, return_tensors="pt", padding=True).to(DEVICE)
                    emb = model.get_text_features(**inputs)

                txt_emb.append(emb.cpu())

            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            txt_times.append(time.perf_counter() - start)

        txt_emb = torch.cat(txt_emb)
        avg_txt_time = (sum(txt_times) / len(txt_times)) * 1000 / BATCH_SIZE

        # Normalize
        img_emb = F.normalize(img_emb, dim=1)
        txt_emb = F.normalize(txt_emb, dim=1)

        # Compute accuracy
        sim = txt_emb @ img_emb.T

        # Text->Image Recall@K
        t2i_r1 = sum(i in torch.topk(sim[i], 1).indices for i in range(len(sim))) / len(sim) * 100
        t2i_r5 = sum(i in torch.topk(sim[i], 5).indices for i in range(len(sim))) / len(sim) * 100
        t2i_r10 = sum(i in torch.topk(sim[i], 10).indices for i in range(len(sim))) / len(sim) * 100

        # Image->Text Recall@K
        sim_i2t = img_emb @ txt_emb.T
        i2t_r1 = sum(i in torch.topk(sim_i2t[i], 1).indices for i in range(len(sim_i2t))) / len(sim_i2t) * 100
        i2t_r5 = sum(i in torch.topk(sim_i2t[i], 5).indices for i in range(len(sim_i2t))) / len(sim_i2t) * 100
        i2t_r10 = sum(i in torch.topk(sim_i2t[i], 10).indices for i in range(len(sim_i2t))) / len(sim_i2t) * 100

        # Compression tolerance
        fp16_emb = img_emb.half().float()
        fp16_emb = F.normalize(fp16_emb, dim=1)
        fp16_sim = F.cosine_similarity(img_emb, fp16_emb).mean().item()
        fp16_drop = (1 - fp16_sim) * 100

        # INT8 simulation
        int8_emb = torch.quantize_per_tensor(img_emb, scale=0.1, zero_point=0, dtype=torch.qint8).dequantize()
        int8_emb = F.normalize(int8_emb, dim=1)
        int8_sim = F.cosine_similarity(img_emb, int8_emb).mean().item()
        int8_drop = (1 - int8_sim) * 100

        print(f"\n📊 Results:")
        print(f"  Image encode: {avg_img_time:.2f}ms")
        print(f"  Text encode:  {avg_txt_time:.2f}ms")
        print(f"  T→I R@10:     {t2i_r10:.1f}%")
        print(f"  I→T R@10:     {i2t_r10:.1f}%")
        print(f"  FP16 drop:    {fp16_drop:.3f}%")
        print(f"  INT8 drop:    {int8_drop:.3f}%")

        results.append({
            'Model': config['name'],
            'Params (M)': f"{total_params/1e6:.1f}",
            'Size (MB)': f"{model_size_mb:.0f}",
            'Img (ms)': f"{avg_img_time:.1f}",
            'Txt (ms)': f"{avg_txt_time:.1f}",
            'T→I R@1': f"{t2i_r1:.1f}",
            'T→I R@5': f"{t2i_r5:.1f}",
            'T→I R@10': f"{t2i_r10:.1f}",
            'I→T R@10': f"{i2t_r10:.1f}",
            'FP16↓': f"{fp16_drop:.3f}",
            'INT8↓': f"{int8_drop:.2f}",
            'Mobile✓': '✓' if avg_img_time < 120 and model_size_mb < 150 else '✗'
        })

        # Cleanup
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"✗ Error: {e}")
        results.append({'Model': config['name'], 'Error': str(e)})

# Final comparison
print("\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Save results
df_results.to_csv('week1_results_flickr30k.csv', index=False)
print("\n✓ Results saved to week1_results_flickr30k.csv")
