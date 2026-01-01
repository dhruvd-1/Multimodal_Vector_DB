"""
WEEK 1 - Embedding Model Evaluation Harness
Samsung Multimodal Vector DB Project

Tests: MobileCLIP-S2, OpenCLIP ViT-S/B-32, SigLIP-S

Usage:
    python week1_model_evaluation.py --model openclip-vit-b
    python week1_model_evaluation.py --all
"""

import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import psutil
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


@dataclass
class EvaluationResults:
    model_name: str
    image_encode_time_ms: float
    text_encode_time_ms: float
    image_batch_throughput: float
    text_batch_throughput: float
    model_size_mb: float
    peak_ram_mb: float
    text_to_image_recall_at_1: float
    text_to_image_recall_at_5: float
    text_to_image_recall_at_10: float
    image_to_text_recall_at_1: float
    image_to_text_recall_at_5: float
    image_to_text_recall_at_10: float
    fp16_accuracy_drop: float
    int8_accuracy_drop: float
    fp16_cosine_similarity: float
    int8_cosine_similarity: float
    android_compatible: bool
    conversion_notes: str


MODELS = {
    "openclip-vit-b": ("openai/clip-vit-base-patch32", 224, 512),
    "openclip-vit-s": ("openai/clip-vit-base-patch16", 224, 512),
}


class ModelEvaluator:
    def __init__(self, model_name, model_id, image_size, emb_dim, device="cpu"):
        self.model_name = model_name
        self.model_id = model_id
        self.image_size = image_size
        self.emb_dim = emb_dim
        self.device = device

    def load(self):
        print(f"Loading {self.model_name}...")
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Loaded")

    def encode_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()

    def encode_texts(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()

    def evaluate(self) -> EvaluationResults:
        print(f"\n{'=' * 70}\nEVALUATING: {self.model_name}\n{'=' * 70}")

        speed = self.measure_speed()
        memory = self.measure_memory()
        accuracy = self.measure_accuracy()
        compression = self.test_compression()
        android = self.check_android()

        return EvaluationResults(
            model_name=self.model_name,
            **speed,
            **memory,
            **accuracy,
            **compression,
            **android,
        )

    def measure_speed(self):
        print("\n--- Speed ---")
        img = Image.new("RGB", (self.image_size, self.image_size))
        txt = "A photo of a cat"

        # Warmup
        for _ in range(5):
            self.encode_images([img])
            self.encode_texts([txt])

        # Image
        times = []
        for _ in range(50):
            start = time.perf_counter()
            self.encode_images([img])
            times.append((time.perf_counter() - start) * 1000)
        img_time = np.mean(times)

        # Text
        times = []
        for _ in range(50):
            start = time.perf_counter()
            self.encode_texts([txt])
            times.append((time.perf_counter() - start) * 1000)
        txt_time = np.mean(times)

        # Throughput
        imgs = [img] * 8
        start = time.perf_counter()
        for _ in range(10):
            self.encode_images(imgs)
        img_thr = 80 / (time.perf_counter() - start)

        txts = [txt] * 8
        start = time.perf_counter()
        for _ in range(10):
            self.encode_texts(txts)
        txt_thr = 80 / (time.perf_counter() - start)

        print(f"  Image: {img_time:.1f}ms | Text: {txt_time:.1f}ms")
        print(f"  Throughput: {img_thr:.1f} img/s, {txt_thr:.1f} txt/s")

        return {
            "image_encode_time_ms": img_time,
            "text_encode_time_ms": txt_time,
            "image_batch_throughput": img_thr,
            "text_batch_throughput": txt_thr,
        }

    def measure_memory(self):
        print("\n--- Memory ---")
        param_size = sum(
            p.nelement() * p.element_size() for p in self.model.parameters()
        )
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        model_mb = (param_size + buffer_size) / (1024**2)

        process = psutil.Process()
        baseline = process.memory_info().rss / (1024**2)

        img = Image.new("RGB", (self.image_size, self.image_size))
        self.encode_images([img] * 32)

        peak = process.memory_info().rss / (1024**2)
        ram_mb = peak - baseline

        print(f"  Model: {model_mb:.1f}MB | RAM: {ram_mb:.1f}MB")

        return {"model_size_mb": model_mb, "peak_ram_mb": ram_mb}

    def measure_accuracy(self, n=100):
        print("\n--- Accuracy (dummy data) ---")
        print("  ⚠️  Replace with Flickr30K/COCO for real eval")

        categories = [
            "cat",
            "dog",
            "car",
            "tree",
            "building",
            "person",
            "food",
            "landscape",
        ]
        images = [
            Image.new(
                "RGB",
                (self.image_size, self.image_size),
                color=(i % 255, (i * 2) % 255, (i * 3) % 255),
            )
            for i in range(n)
        ]
        texts = [f"A photo of a {categories[i % len(categories)]}" for i in range(n)]

        img_embs = self.encode_images(images)
        txt_embs = self.encode_texts(texts)

        t2i_sim = txt_embs @ img_embs.T
        i2t_sim = img_embs @ txt_embs.T

        def recall_at_k(sim, k):
            correct = sum(
                1 for i in range(len(sim)) if i in np.argsort(sim[i])[::-1][:k]
            )
            return correct / len(sim)

        t2i_r1 = recall_at_k(t2i_sim, 1)
        t2i_r5 = recall_at_k(t2i_sim, 5)
        t2i_r10 = recall_at_k(t2i_sim, 10)
        i2t_r1 = recall_at_k(i2t_sim, 1)
        i2t_r5 = recall_at_k(i2t_sim, 5)
        i2t_r10 = recall_at_k(i2t_sim, 10)

        print(f"  T→I R@10: {t2i_r10:.4f} | I→T R@10: {i2t_r10:.4f}")

        return {
            "text_to_image_recall_at_1": t2i_r1,
            "text_to_image_recall_at_5": t2i_r5,
            "text_to_image_recall_at_10": t2i_r10,
            "image_to_text_recall_at_1": i2t_r1,
            "image_to_text_recall_at_5": i2t_r5,
            "image_to_text_recall_at_10": i2t_r10,
        }

    def test_compression(self):
        print("\n--- Compression ---")
        img = Image.new("RGB", (self.image_size, self.image_size))
        embs_fp32 = self.encode_images([img] * 10)

        # FP16
        embs_fp16 = embs_fp32.astype(np.float16).astype(np.float32)
        fp16_sim = np.mean(
            [np.dot(embs_fp32[i], embs_fp16[i]) for i in range(len(embs_fp32))]
        )

        # INT8
        scale = 127.0 / np.max(np.abs(embs_fp32))
        embs_int8 = (
            np.round(embs_fp32 * scale).astype(np.int8).astype(np.float32)
        ) / scale
        int8_sim = np.mean(
            [np.dot(embs_fp32[i], embs_int8[i]) for i in range(len(embs_fp32))]
        )

        fp16_drop = 1.0 - fp16_sim
        int8_drop = 1.0 - int8_sim

        print(f"  FP16: {fp16_sim:.6f} (drop: {fp16_drop:.4%})")
        print(f"  INT8: {int8_sim:.6f} (drop: {int8_drop:.4%})")

        if fp16_drop < 0.01:
            print("  ✓ FP16 OK")
        if int8_drop < 0.05:
            print("  ✓ INT8 OK")

        return {
            "fp16_cosine_similarity": float(fp16_sim),
            "int8_cosine_similarity": float(int8_sim),
            "fp16_accuracy_drop": float(fp16_drop),
            "int8_accuracy_drop": float(int8_drop),
        }

    def check_android(self):
        print("\n--- Android ---")
        notes = []
        compatible = True

        size = self.measure_memory()["model_size_mb"]
        if size > 150:
            notes.append(f"⚠️  Large: {size:.0f}MB")
            compatible = False
        else:
            notes.append(f"✓ Size OK: {size:.0f}MB")

        try:
            dummy = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
            torch.jit.trace(self.model.vision_model, dummy)
            notes.append("✓ TorchScript OK")
        except Exception as e:
            notes.append(f"✗ TorchScript failed")
            compatible = False

        for note in notes:
            print(f"  {note}")

        return {
            "android_compatible": compatible,
            "conversion_notes": " | ".join(notes),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    print("=" * 70)
    print("WEEK 1 — MODEL EVALUATION")
    print("Samsung Multimodal Vector DB")
    print("=" * 70)

    models_to_test = list(MODELS.keys()) if args.model == "all" else [args.model]

    results = []
    for key in models_to_test:
        model_id, img_size, emb_dim = MODELS[key]
        evaluator = ModelEvaluator(key, model_id, img_size, emb_dim, args.device)
        evaluator.load()
        result = evaluator.evaluate()
        results.append(result)

    # Save
    output = {
        "models": [asdict(r) for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved to evaluation_results.json")

    # Compare
    print(f"\n{'=' * 70}\nCOMPARISON\n{'=' * 70}")
    print(
        f"{'Model':<20} {'Img(ms)':<10} {'T→I R@10':<12} {'FP16 Drop':<12} {'Size(MB)':<10}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r.model_name:<20} {r.image_encode_time_ms:<10.1f} "
            f"{r.text_to_image_recall_at_10:<12.4f} {r.fp16_accuracy_drop:<12.4%} "
            f"{r.model_size_mb:<10.1f}"
        )

    best_speed = min(results, key=lambda x: x.image_encode_time_ms)
    print(
        f"\n⚡ Fastest: {best_speed.model_name} ({best_speed.image_encode_time_ms:.1f}ms)"
    )

    print(f"\n{'=' * 70}\n✓ COMPLETE\n{'=' * 70}")
    print("\nNext:")
    print("1. Get Flickr30K/COCO for real accuracy")
    print("2. Test on Android device")
    print("3. Week 2: Fine-tuning")


if __name__ == "__main__":
    main()
