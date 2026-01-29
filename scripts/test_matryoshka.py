"""
Low-level test script for Matryoshka Projection logic.
Focuses on neural network layers and weight loading.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedders.projection import MatryoshkaProjection, MatryoshkaLoss

def test_weight_loading():
    """Test loading trained weights into the projection layer."""
    print("\n--- Testing Weight Loading ---")
    
    weights_path = os.path.join("models", "matryoshka_final.pt")
    
    # Initialize projection
    proj = MatryoshkaProjection(input_dim=512, matryoshka_dims=[512, 256, 128, 64])
    
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        # Use class method to load
        loaded_proj = MatryoshkaProjection.load(weights_path)
        
        assert isinstance(loaded_proj, MatryoshkaProjection)
        assert loaded_proj.input_dim == 512
        print("✓ Weights loaded successfully via class method")
        
        # Verify layer structure
        assert hasattr(loaded_proj, 'projection')
        assert hasattr(loaded_proj, 'dim_scales')
        print("✓ Layer structure verified")
    else:
        print(f"Skipping weight load test: {weights_path} not found.")

def test_layer_consistency():
    """Test that the layer produces consistent results."""
    print("\n--- Testing Layer Consistency ---")
    
    proj = MatryoshkaProjection(input_dim=512)
    proj.eval()
    
    x = torch.randn(1, 512)
    
    # Forward pass multiple times
    with torch.no_grad():
        out1 = proj(x, output_dim=128)
        out2 = proj(x, output_dim=128)
        
    assert torch.allclose(out1, out2), "Projection is not deterministic!"
    print("✓ Consistency passed (Same input = Same output)")

def test_dimension_scaling():
    """Test that dimension-specific scaling is applied."""
    print("\n--- Testing Dimension Scaling ---")
    
    proj = MatryoshkaProjection(input_dim=512)
    
    # Check that dim_scales contains the expected keys
    for dim in [512, 256, 128, 64]:
        assert str(dim) in proj.dim_scales
        
    print("✓ Dimension-specific scaling factors initialized")

def main():
    """Run low-level tests."""
    print("=" * 60)
    print("RUNNING MATRYOSHKA LOW-LEVEL TESTS")
    print("=" * 60)
    
    try:
        test_weight_loading()
        test_layer_consistency()
        test_dimension_scaling()
        
        print("\n" + "=" * 60)
        print("ALL LOW-LEVEL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()