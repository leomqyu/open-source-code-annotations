"""
Test block for BasicLayer.forward branch handling tuple vs single-value outputs.

This test covers the branch at lines 576-581 in swin_transformer_moe.py:
    if isinstance(out, tuple):
        x = out[0]
        cur_l_aux = out[1]
        l_aux = cur_l_aux + l_aux
    else:
        x = out
"""

import torch
import torch.nn as nn
import sys
import os

# Add the models directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

# Check if tutel is available for MoE tests
try:
    from tutel import moe as tutel_moe
    TUTEL_AVAILABLE = True
except ImportError:
    TUTEL_AVAILABLE = False
    print("Warning: Tutel not available. MoE tests will be skipped.")

from swin_transformer_moe import BasicLayer, SwinTransformerBlock


class MockBlockReturningTuple(nn.Module):
    """Mock block that returns a tuple (x, l_aux) like MoE blocks."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        # Return tuple like MoE blocks do
        l_aux = torch.tensor(0.5, requires_grad=True)
        return x, l_aux


class MockBlockReturningSingle(nn.Module):
    """Mock block that returns a single value like regular blocks."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        
    def forward(self, x):
        # Return single value like regular blocks do
        return self.linear(x)


def test_basic_layer_tuple_branch():
    """Test BasicLayer forward when blocks return tuples (MoE blocks)."""
    if not TUTEL_AVAILABLE:
        print("  ⚠ Skipping tuple branch test (Tutel not available)")
        return
    
    print("Testing tuple branch (MoE blocks)...")
    
    dim = 96
    input_resolution = (56, 56)
    depth = 2
    
    # Create BasicLayer with MoE blocks (which return tuples)
    layer = BasicLayer(
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=3,
        window_size=7,
        moe_block=[0, 1],  # All blocks are MoE blocks
        num_local_experts=2,
        top_value=2
    )
    
    # Create input tensor
    B, H, W = 2, 56, 56
    x = torch.randn(B, H * W, dim)
    
    # Forward pass
    output, l_aux = layer(x)
    
    # Assertions
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert isinstance(l_aux, torch.Tensor), "l_aux should be a tensor"
    assert output.shape == (B, (H // 2) * (W // 2), dim * 2), "Output shape should match after downsample"
    assert l_aux.item() > 0, "l_aux should be accumulated from MoE blocks"
    
    print(f"  ✓ Tuple branch test passed")
    print(f"    Output shape: {output.shape}")
    print(f"    l_aux value: {l_aux.item():.4f}")


def test_basic_layer_single_branch():
    """Test BasicLayer forward when blocks return single values (regular blocks)."""
    print("Testing single value branch (regular blocks)...")
    
    dim = 96
    input_resolution = (56, 56)
    depth = 2
    
    # Create BasicLayer with regular blocks (which return single values)
    layer = BasicLayer(
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=3,
        window_size=7,
        moe_block=[-1],  # No MoE blocks (all regular)
        num_local_experts=1,
        top_value=1
    )
    
    # Create input tensor
    B, H, W = 2, 56, 56
    x = torch.randn(B, H * W, dim)
    
    # Forward pass
    output, l_aux = layer(x)
    
    # Assertions
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert isinstance(l_aux, torch.Tensor), "l_aux should be a tensor"
    assert output.shape == (B, (H // 2) * (W // 2), dim * 2), "Output shape should match after downsample"
    assert l_aux.item() == 0.0, "l_aux should be 0.0 for regular blocks"
    
    print(f"  ✓ Single value branch test passed")
    print(f"    Output shape: {output.shape}")
    print(f"    l_aux value: {l_aux.item():.4f}")


def test_basic_layer_mixed_branch():
    """Test BasicLayer forward with mixed MoE and regular blocks."""
    if not TUTEL_AVAILABLE:
        print("  ⚠ Skipping mixed blocks test (Tutel not available)")
        return
    
    print("Testing mixed blocks (both tuple and single value branches)...")
    
    dim = 96
    input_resolution = (56, 56)
    depth = 4
    
    # Create BasicLayer with mixed blocks (some MoE, some regular)
    layer = BasicLayer(
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=3,
        window_size=7,
        moe_block=[0, 2],  # Blocks 0 and 2 are MoE (return tuples), 1 and 3 are regular
        num_local_experts=2,
        top_value=2
    )
    
    # Create input tensor
    B, H, W = 2, 56, 56
    x = torch.randn(B, H * W, dim)
    
    # Forward pass
    output, l_aux = layer(x)
    
    # Assertions
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert isinstance(l_aux, torch.Tensor), "l_aux should be a tensor"
    assert output.shape == (B, (H // 2) * (W // 2), dim * 2), "Output shape should match after downsample"
    assert l_aux.item() > 0, "l_aux should be accumulated from MoE blocks"
    
    print(f"  ✓ Mixed blocks test passed")
    print(f"    Output shape: {output.shape}")
    print(f"    l_aux value: {l_aux.item():.4f}")


def test_basic_layer_with_checkpoint():
    """Test BasicLayer forward with checkpointing enabled (both branches)."""
    if not TUTEL_AVAILABLE:
        print("  ⚠ Skipping checkpointing test (Tutel not available)")
        return
    
    print("Testing with checkpointing enabled...")
    
    dim = 96
    input_resolution = (56, 56)
    depth = 2
    
    # Create BasicLayer with checkpointing and MoE blocks
    layer = BasicLayer(
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=3,
        window_size=7,
        moe_block=[0, 1],
        num_local_experts=2,
        top_value=2,
        use_checkpoint=True
    )
    
    # Create input tensor
    B, H, W = 2, 56, 56
    x = torch.randn(B, H * W, dim)
    
    # Forward pass
    output, l_aux = layer(x)
    
    # Assertions
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert isinstance(l_aux, torch.Tensor), "l_aux should be a tensor"
    assert output.shape == (B, (H // 2) * (W // 2), dim * 2), "Output shape should match"
    assert l_aux.item() > 0, "l_aux should be accumulated"
    
    print(f"  ✓ Checkpointing test passed")
    print(f"    Output shape: {output.shape}")
    print(f"    l_aux value: {l_aux.item():.4f}")


def test_basic_layer_no_downsample():
    """Test BasicLayer forward without downsample layer."""
    if not TUTEL_AVAILABLE:
        print("  ⚠ Skipping no downsample test (Tutel not available)")
        return
    
    print("Testing without downsample layer...")
    
    dim = 96
    input_resolution = (56, 56)
    depth = 2
    
    # Create BasicLayer without downsample (last layer scenario)
    layer = BasicLayer(
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=3,
        window_size=7,
        moe_block=[0, 1],
        num_local_experts=2,
        top_value=2,
        downsample=None  # No downsample
    )
    
    # Create input tensor
    B, H, W = 2, 56, 56
    x = torch.randn(B, H * W, dim)
    
    # Forward pass
    output, l_aux = layer(x)
    
    # Assertions
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert isinstance(l_aux, torch.Tensor), "l_aux should be a tensor"
    assert output.shape == (B, H * W, dim), "Output shape should match input resolution"
    assert l_aux.item() > 0, "l_aux should be accumulated"
    
    print(f"  ✓ No downsample test passed")
    print(f"    Output shape: {output.shape}")
    print(f"    l_aux value: {l_aux.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing BasicLayer.forward branch (tuple vs single value)")
    print("=" * 60)
    print()
    
    try:
        test_basic_layer_tuple_branch()
        print()
        
        test_basic_layer_single_branch()
        print()
        
        test_basic_layer_mixed_branch()
        print()
        
        test_basic_layer_with_checkpoint()
        print()
        
        test_basic_layer_no_downsample()
        print()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

