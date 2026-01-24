"""
Matryoshka Projection Training Script for Kaggle.

This script trains a Matryoshka projection layer on top of frozen CLIP embeddings.
The trained projection enables truncatable embeddings (512D → 256D → 128D → 64D)
while preserving semantic quality at each dimension.

Dataset: COCO Captions (Kaggle: nikhil7280/coco-image-caption)
Runtime: GPU T4/P100, ~2-4 hours
Output: matryoshka_final.pt

Kaggle Setup:
1. Create new notebook on Kaggle
2. Add dataset: Search "COCO Captions" → Add `nikhil7280/coco-image-caption`
3. Enable GPU: Settings → Accelerator → GPU T4 x2
4. Copy this script and run
5. Download `matryoshka_final.pt` when done

Usage after training:
    from src.embedders.text_embedder import TextEmbedder

    embedder = TextEmbedder(use_matryoshka=True, matryoshka_dim=256)
    embedder.load_model()
    embedder.load_matryoshka_weights('matryoshka_final.pt')

    emb = embedder.embed("cat playing", output_dim=128)  # Get 128D embedding
"""

import os
import json
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Check for transformers
try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    print("Installing transformers...")
    os.system("pip install transformers")
    from transformers import CLIPModel, CLIPProcessor

# Check for PIL
try:
    from PIL import Image
except ImportError:
    print("Installing Pillow...")
    os.system("pip install Pillow")
    from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Training configuration."""
    # Model
    CLIP_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    INPUT_DIM = 512
    MATRYOSHKA_DIMS = [512, 256, 128, 64]

    # Training
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500

    # Data
    MAX_SAMPLES = 100000  # Use subset for faster training (None = all)
    NUM_WORKERS = 2

    # Paths (Kaggle)
    ANNOTATIONS_FILE = '/kaggle/input/coco-image-caption/annotations/captions_train2017.json'
    IMAGES_DIR = '/kaggle/input/coco-image-caption/train2017'

    # Output
    CHECKPOINT_DIR = './checkpoints'
    FINAL_MODEL_PATH = './matryoshka_final.pt'


# ============================================================
# MATRYOSHKA PROJECTION
# ============================================================

class MatryoshkaProjection(nn.Module):
    """
    Matryoshka Representation Learning projection layer.

    Produces embeddings that can be truncated to smaller dimensions
    while preserving semantic quality.
    """

    def __init__(
        self,
        input_dim: int = 512,
        matryoshka_dims: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.matryoshka_dims = matryoshka_dims or [512, 256, 128, 64]
        self.max_dim = max(self.matryoshka_dims)
        self.dropout = dropout

        # MLP projection
        hidden_dim = input_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.max_dim),
        )
        self.layer_norm = nn.LayerNorm(self.max_dim)

        # Learnable scales per dimension
        self.dim_scales = nn.ParameterDict({
            str(dim): nn.Parameter(torch.ones(1))
            for dim in self.matryoshka_dims
        })

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, output_dim: Optional[int] = None) -> torch.Tensor:
        """Forward pass with optional dimension truncation."""
        projected = self.projection(x)
        projected = self.layer_norm(projected)

        if output_dim is not None:
            projected = projected[:, :output_dim]
            if str(output_dim) in self.dim_scales:
                projected = projected * self.dim_scales[str(output_dim)]

        return F.normalize(projected, p=2, dim=-1)

    def forward_multi_scale(self, x: torch.Tensor) -> dict:
        """Get embeddings at all Matryoshka scales."""
        full = self.projection(x)
        full = self.layer_norm(full)

        results = {}
        for dim in self.matryoshka_dims:
            truncated = full[:, :dim] * self.dim_scales[str(dim)]
            results[dim] = F.normalize(truncated, p=2, dim=-1)
        return results


# ============================================================
# MATRYOSHKA LOSS
# ============================================================

class MatryoshkaLoss(nn.Module):
    """
    Multi-scale contrastive loss for Matryoshka training.

    Computes InfoNCE loss at each dimension scale and combines them
    with weights favoring larger dimensions.
    """

    def __init__(
        self,
        matryoshka_dims: List[int] = None,
        temperature: float = 0.07
    ):
        super().__init__()
        self.dims = matryoshka_dims or [512, 256, 128, 64]
        self.temperature = temperature

        # Weight larger dims more (1/1, 1/2, 1/3, 1/4, ...)
        weights = [1.0 / (i + 1) for i in range(len(self.dims))]
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def forward(
        self,
        image_embeds_dict: dict,
        text_embeds_dict: dict
    ) -> tuple:
        """Compute multi-scale contrastive loss."""
        total_loss = 0.0
        loss_dict = {}

        for dim, weight in zip(self.dims, self.weights):
            img_emb = image_embeds_dict[dim]
            txt_emb = text_embeds_dict[dim]

            # Contrastive loss (InfoNCE)
            logits = torch.matmul(img_emb, txt_emb.T) / self.temperature
            labels = torch.arange(len(logits), device=logits.device)

            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)

            dim_loss = (loss_i2t + loss_t2i) / 2
            total_loss += weight * dim_loss
            loss_dict[f'loss_{dim}d'] = dim_loss.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict


# ============================================================
# DATASET
# ============================================================

class COCOCaptionsDataset(Dataset):
    """COCO Captions dataset for image-text pairs."""

    def __init__(
        self,
        annotations_file: str,
        images_dir: str,
        processor,
        max_samples: Optional[int] = None
    ):
        print(f"Loading annotations from {annotations_file}")
        with open(annotations_file, 'r') as f:
            data = json.load(f)

        # Build image_id -> filename mapping
        id_to_file = {img['id']: img['file_name'] for img in data['images']}

        # Get image-caption pairs
        self.pairs = []
        for ann in data['annotations']:
            img_file = id_to_file.get(ann['image_id'])
            if img_file:
                img_path = os.path.join(images_dir, img_file)
                if os.path.exists(img_path):
                    self.pairs.append({
                        'image_path': img_path,
                        'caption': ann['caption']
                    })

        if max_samples:
            self.pairs = self.pairs[:max_samples]

        self.processor = processor
        print(f"Loaded {len(self.pairs)} image-caption pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        try:
            image = Image.open(pair['image_path']).convert('RGB')
            inputs = self.processor(
                images=image,
                text=pair['caption'],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77
            )

            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
            }
        except Exception as e:
            # Return a dummy sample on error
            print(f"Error loading {pair['image_path']}: {e}")
            return self.__getitem__((idx + 1) % len(self))


# ============================================================
# TRAINING
# ============================================================

def train_matryoshka(config: Config = None):
    """Main training function."""
    config = config or Config()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load CLIP (frozen)
    # --------------------------------------------------------
    print(f"Loading CLIP model: {config.CLIP_MODEL}")
    clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
    clip_model.to(device)
    clip_model.eval()

    # Freeze CLIP
    for param in clip_model.parameters():
        param.requires_grad = False

    print("CLIP model loaded and frozen")

    # --------------------------------------------------------
    # Initialize Matryoshka projection (trainable)
    # --------------------------------------------------------
    matryoshka = MatryoshkaProjection(
        input_dim=config.INPUT_DIM,
        matryoshka_dims=config.MATRYOSHKA_DIMS,
    ).to(device)

    criterion = MatryoshkaLoss(matryoshka_dims=config.MATRYOSHKA_DIMS)

    optimizer = torch.optim.AdamW(
        matryoshka.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS
    )

    print(f"Matryoshka projection initialized: dims={config.MATRYOSHKA_DIMS}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    dataset = COCOCaptionsDataset(
        annotations_file=config.ANNOTATIONS_FILE,
        images_dir=config.IMAGES_DIR,
        processor=processor,
        max_samples=config.MAX_SAMPLES
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    best_loss = float('inf')

    for epoch in range(config.EPOCHS):
        matryoshka.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

        for batch in pbar:
            # Move to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get CLIP embeddings (frozen, no grad)
            with torch.no_grad():
                image_embeds = clip_model.get_image_features(pixel_values)
                text_embeds = clip_model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Normalize CLIP outputs
                image_embeds = F.normalize(image_embeds, p=2, dim=-1)
                text_embeds = F.normalize(text_embeds, p=2, dim=-1)

            # Project through Matryoshka (trainable)
            image_multi = matryoshka.forward_multi_scale(image_embeds)
            text_multi = matryoshka.forward_multi_scale(text_embeds)

            # Compute loss
            loss, loss_dict = criterion(image_multi, text_multi)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(matryoshka.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        scheduler.step()

        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR,
            f'matryoshka_epoch_{epoch+1}.pt'
        )
        torch.save({
            'epoch': epoch + 1,
            'state_dict': matryoshka.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': avg_loss,
            'config': {
                'input_dim': config.INPUT_DIM,
                'matryoshka_dims': config.MATRYOSHKA_DIMS,
            }
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'state_dict': matryoshka.state_dict(),
                'input_dim': config.INPUT_DIM,
                'matryoshka_dims': config.MATRYOSHKA_DIMS,
                'max_dim': max(config.MATRYOSHKA_DIMS),
                'use_mlp': True,
                'dropout': 0.1,
            }, config.FINAL_MODEL_PATH)
            print(f"Best model saved: {config.FINAL_MODEL_PATH}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model: {config.FINAL_MODEL_PATH}")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 60)

    return matryoshka


# ============================================================
# EVALUATION
# ============================================================

def evaluate_matryoshka(model_path: str, device: str = 'cuda'):
    """Quick evaluation of trained Matryoshka model."""
    print(f"\nEvaluating: {model_path}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)

    matryoshka = MatryoshkaProjection(
        input_dim=checkpoint.get('input_dim', 512),
        matryoshka_dims=checkpoint.get('matryoshka_dims', [512, 256, 128, 64]),
    ).to(device)
    matryoshka.load_state_dict(checkpoint['state_dict'])
    matryoshka.eval()

    # Test with random embeddings
    with torch.no_grad():
        test_input = torch.randn(4, 512).to(device)
        test_input = F.normalize(test_input, p=2, dim=-1)

        multi_scale = matryoshka.forward_multi_scale(test_input)

        print("\nOutput dimensions:")
        for dim, emb in multi_scale.items():
            norm = torch.norm(emb, dim=-1).mean().item()
            print(f"  {dim}D: shape={list(emb.shape)}, avg_norm={norm:.4f}")

    print("\nModel evaluation complete!")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Matryoshka Projection')
    parser.add_argument('--evaluate', type=str, help='Path to model to evaluate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=100000, help='Max samples')
    args = parser.parse_args()

    if args.evaluate:
        evaluate_matryoshka(args.evaluate)
    else:
        # Update config from args
        Config.EPOCHS = args.epochs
        Config.BATCH_SIZE = args.batch_size
        Config.MAX_SAMPLES = args.max_samples

        # Train
        train_matryoshka()
