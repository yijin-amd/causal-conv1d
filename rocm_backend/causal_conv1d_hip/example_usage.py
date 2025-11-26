#!/usr/bin/env python3
"""
Example usage of Causal Conv1D HIP implementation
Demonstrates how to use the HIP version similar to CUDA version
"""

import torch
import time
import sys
import os

# Add current directory to path for importing the interface
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from causal_conv1d_hip_interface import (
        causal_conv1d_hip_fn,
        causal_conv1d_hip_ref,
        test_causal_conv1d_hip,
    )
    HIP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import HIP interface: {e}")
    HIP_AVAILABLE = False


def example_basic_usage():
    """Example 1: Basic usage"""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    if not HIP_AVAILABLE:
        print("HIP extension not available. Skipping...")
        return
    
    # Configuration
    batch, dim, seqlen, width = 2, 64, 512, 4
    device = 'cuda'  # In ROCm, 'cuda' maps to HIP device
    
    print(f"Configuration:")
    print(f"  Batch size: {batch}")
    print(f"  Dimensions: {dim}")
    print(f"  Sequence length: {seqlen}")
    print(f"  Kernel width: {width}")
    
    # Create test data
    torch.manual_seed(42)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    
    print(f"\nTensor shapes:")
    print(f"  x:      {x.shape}")
    print(f"  weight: {weight.shape}")
    print(f"  bias:   {bias.shape}")
    
    # Run HIP implementation
    print(f"\nRunning HIP implementation...")
    out = causal_conv1d_hip_fn(x, weight, bias, activation='silu')
    
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"  Output mean:  {out.mean().item():.4f}")
    print(f"  Output std:   {out.std().item():.4f}")
    
    print("\n✓ Basic usage completed successfully!")


def example_compare_with_reference():
    """Example 2: Compare with reference implementation"""
    print("\n" + "="*60)
    print("Example 2: Compare with Reference Implementation")
    print("="*60)
    
    if not HIP_AVAILABLE:
        print("HIP extension not available. Skipping...")
        return
    
    # Configuration
    batch, dim, seqlen, width = 4, 128, 1024, 4
    device = 'cuda'
    
    print(f"Configuration: batch={batch}, dim={dim}, seqlen={seqlen}, width={width}")
    
    # Create test data
    torch.manual_seed(42)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    
    # Test different configurations
    configs = [
        ("No activation, with bias", None, True),
        ("SiLU activation, with bias", "silu", True),
        ("SiLU activation, no bias", "silu", False),
        ("Swish activation, with bias", "swish", True),
    ]
    
    for name, activation, use_bias in configs:
        print(f"\n[{name}]")
        
        b = bias if use_bias else None
        
        # HIP implementation
        out_hip = causal_conv1d_hip_fn(x, weight, b, activation=activation)
        
        # Reference implementation
        out_ref = causal_conv1d_hip_ref(x, weight, b, activation=activation)
        
        # Compare
        diff = (out_hip - out_ref).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  Max difference:  {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print(f"  Status: ✓ PASS")
        else:
            print(f"  Status: ✗ FAIL")
    
    print("\n✓ Comparison completed!")


def example_performance_benchmark():
    """Example 3: Performance benchmark"""
    print("\n" + "="*60)
    print("Example 3: Performance Benchmark")
    print("="*60)
    
    if not HIP_AVAILABLE:
        print("HIP extension not available. Skipping...")
        return
    
    # Test different sizes
    configs = [
        ("Small",  2, 32,  512,   4),
        ("Medium", 4, 64,  2048,  4),
        ("Large",  8, 128, 4096,  4),
    ]
    
    device = 'cuda'
    num_warmup = 10
    num_iters = 100
    
    print(f"Warmup iterations: {num_warmup}")
    print(f"Benchmark iterations: {num_iters}")
    
    for name, batch, dim, seqlen, width in configs:
        print(f"\n[{name}] batch={batch}, dim={dim}, seqlen={seqlen}, width={width}")
        
        # Create test data
        x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
        weight = torch.randn(dim, width, device=device, dtype=torch.float32)
        bias = torch.randn(dim, device=device, dtype=torch.float32)
        
        # Warmup
        for _ in range(num_warmup):
            _ = causal_conv1d_hip_fn(x, weight, bias, activation='silu')
        torch.cuda.synchronize()
        
        # Benchmark HIP implementation
        start = time.time()
        for _ in range(num_iters):
            out = causal_conv1d_hip_fn(x, weight, bias, activation='silu')
        torch.cuda.synchronize()
        hip_time = (time.time() - start) / num_iters * 1000  # ms
        
        # Benchmark reference implementation
        start = time.time()
        for _ in range(num_iters):
            out_ref = causal_conv1d_hip_ref(x, weight, bias, activation='silu')
        torch.cuda.synchronize()
        ref_time = (time.time() - start) / num_iters * 1000  # ms
        
        # Calculate throughput
        total_elements = batch * dim * seqlen
        hip_throughput = total_elements / (hip_time / 1000) / 1e9  # GFLOPS
        ref_throughput = total_elements / (ref_time / 1000) / 1e9  # GFLOPS
        
        print(f"  HIP implementation: {hip_time:.3f} ms ({hip_throughput:.2f} GFLOPS)")
        print(f"  Reference impl:     {ref_time:.3f} ms ({ref_throughput:.2f} GFLOPS)")
        print(f"  Speedup:            {ref_time/hip_time:.2f}x")
    
    print("\n✓ Performance benchmark completed!")


def example_different_widths():
    """Example 4: Test different kernel widths"""
    print("\n" + "="*60)
    print("Example 4: Different Kernel Widths")
    print("="*60)
    
    if not HIP_AVAILABLE:
        print("HIP extension not available. Skipping...")
        return
    
    batch, dim, seqlen = 2, 64, 512
    device = 'cuda'
    
    # Create test data
    torch.manual_seed(42)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    
    for width in [2, 3, 4]:
        print(f"\n[Width = {width}]")
        
        weight = torch.randn(dim, width, device=device, dtype=torch.float32)
        
        # HIP implementation
        out_hip = causal_conv1d_hip_fn(x, weight, bias, activation='silu')
        
        # Reference implementation
        out_ref = causal_conv1d_hip_ref(x, weight, bias, activation='silu')
        
        # Compare
        diff = (out_hip - out_ref).abs()
        max_diff = diff.max().item()
        
        print(f"  Output shape:    {out_hip.shape}")
        print(f"  Max difference:  {max_diff:.6f}")
        print(f"  Status:          {'✓ PASS' if max_diff < 1e-3 else '✗ FAIL'}")
    
    print("\n✓ Width test completed!")


def main():
    """Run all examples"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*15 + "Causal Conv1D HIP Examples" + " "*17 + "║")
    print("╚" + "="*58 + "╝")
    
    if not HIP_AVAILABLE:
        print("\n⚠️  HIP extension not available!")
        print("Please compile the extension first:")
        print("  cd /workspace/causal-conv1d/rocm_backend/hip_backend/fwd")
        print("  ./compile_hip_extension.sh")
        return 1
    
    # Check if CUDA/HIP is available
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA/HIP device not available!")
        return 1
    
    print(f"\n✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA/HIP available: {torch.cuda.is_available()}")
    print(f"✓ Device count: {torch.cuda.device_count()}")
    print(f"✓ Current device: {torch.cuda.current_device()}")
    print(f"✓ Device name: {torch.cuda.get_device_name(0)}")
    
    try:
        # Run examples
        example_basic_usage()
        example_compare_with_reference()
        example_different_widths()
        example_performance_benchmark()
        
        # Run comprehensive tests
        print("\n" + "="*60)
        print("Running Comprehensive Tests")
        print("="*60)
        test_causal_conv1d_hip()
        
        print("\n" + "╔" + "="*58 + "╗")
        print("║" + " "*17 + "✅ All Examples Completed!" + " "*17 + "║")
        print("╚" + "="*58 + "╝\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

