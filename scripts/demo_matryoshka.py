"""
Demo script showing Matryoshka embeddings at different dimensions.
Tests the trained model with sample embeddings.
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedders.projection import MatryoshkaProjection

def demo_matryoshka():
    """Demonstrate Matryoshka embeddings at multiple scales."""
    print("=" * 60)
    print("MATRYOSHKA EMBEDDINGS DEMO")
    print("=" * 60)
    
    # Load trained model
    model_path = os.path.join("models", "matryoshka_final.pt")
    proj = MatryoshkaProjection.load(model_path)
    
    print(f"\nModel: {proj}")
    print(f"Supported dimensions: {proj.matryoshka_dims}")
    
    # Create sample embeddings (simulating CLIP outputs)
    print("\n--- Creating sample embeddings ---")
    batch_size = 3
    sample_embeddings = torch.randn(batch_size, proj.input_dim)
    
    # Normalize like real CLIP embeddings
    sample_embeddings = torch.nn.functional.normalize(sample_embeddings, p=2, dim=-1)
    print(f"Sample input shape: {sample_embeddings.shape}")
    
    # Test at different dimensions
    print("\n--- Testing at different dimensions ---")
    with torch.no_grad():
        for dim in proj.matryoshka_dims:
            output = proj(sample_embeddings, output_dim=dim)
            print(f"  {dim}D output: shape={output.shape}, "
                  f"norm={output[0].norm().item():.4f}")
    
    # Test multi-scale output
    print("\n--- Multi-scale output ---")
    with torch.no_grad():
        multi_outputs = proj.forward_multi_scale(sample_embeddings)
        
    for dim, embs in sorted(multi_outputs.items(), reverse=True):
        print(f"  {dim}D: {embs.shape}")
    
    # Test device recommendations
    print("\n--- Device-specific dimension recommendations ---")
    
    scenarios = [
        ("High-end server", {"memory_mb": 4096, "latency_ms": 10}),
        ("Mid-range device", {"memory_mb": 512, "latency_ms": 5}),
        ("Mobile device", {"memory_mb": 128, "latency_ms": 2}),
        ("IoT device", {"memory_mb": 32, "latency_ms": 1}),
    ]
    
    for name, constraints in scenarios:
        optimal_dim = proj.get_optimal_dim_for_device(**constraints)
        print(f"  {name:20s} -> {optimal_dim}D "
              f"(memory={constraints['memory_mb']}MB, latency={constraints['latency_ms']}ms)")
    
    # Test nested property
    print("\n--- Verifying nested embedding property ---")
    with torch.no_grad():
        full_512 = proj(sample_embeddings[0:1], output_dim=512)
        truncated_256 = full_512[:, :256]
        direct_256 = proj(sample_embeddings[0:1], output_dim=256)
        
        # Note: They won't be exactly equal due to dimension-specific scaling
        # but the first 256 dimensions of 512D should be similar to 256D
        print(f"  Full 512D (first 256): {truncated_256[0, :5].numpy()}")
        print(f"  Direct 256D (first 5):  {direct_256[0, :5].numpy()}")
        print(f"  Cosine similarity between truncated and direct 256D: "
              f"{torch.cosine_similarity(truncated_256, direct_256).item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ DEMO COMPLETE")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  • Matryoshka embeddings support multiple dimensions: 512, 256, 128, 64")
    print("  • All outputs are L2-normalized for cosine similarity")
    print("  • Smaller dimensions suitable for resource-constrained devices")
    print("  • Nested property allows flexibility in deployment")

if __name__ == "__main__":
    demo_matryoshka()
