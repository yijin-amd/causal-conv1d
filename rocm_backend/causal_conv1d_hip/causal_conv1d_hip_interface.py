# Copyright (c) 2024, Tri Dao.
# Adapted for ROCm/HIP

import torch
import torch.nn.functional as F

# Import the HIP extension
try:
    import causal_conv1d_hip_ext
except ImportError:
    causal_conv1d_hip_ext = None
    print("Warning: causal_conv1d_hip_ext not found. Please compile the HIP extension.")


class CausalConv1dHIPFn(torch.autograd.Function):
    """
    HIP implementation of Causal Conv1D
    Similar to CausalConv1dFn but uses HIP backend
    """
    
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        activation=None,
    ):
        """
        Forward pass for causal conv1d using HIP backend
        
        Args:
            x: (batch, dim, seqlen) - input tensor
            weight: (dim, width) - convolution weights
            bias: (dim,) - optional bias
            activation: None, "silu", or "swish"
        
        Returns:
            out: (batch, dim, seqlen) - output tensor
        """
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        
        # Ensure tensors are contiguous
        if x.stride(2) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        
        # Make sure x is channel-first: (batch, dim, seqlen) with stride[2] == 1
        if x.stride(1) == 1:
            # Channel-last, convert to channel-first
            x = x.transpose(1, 2).contiguous()
        
        bias = bias.contiguous() if bias is not None else None
        
        ctx.activation = activation in ["silu", "swish"]
        
        # Call HIP extension
        if causal_conv1d_hip_ext is None:
            raise RuntimeError("HIP extension not available")
        
        out = causal_conv1d_hip_ext.causal_conv1d_fwd_hip(
            x, weight, bias, ctx.activation
        )
        
        ctx.save_for_backward(x, weight, bias)
        return out
    
    @staticmethod
    def backward(ctx, dout):
        """
        Backward pass (not implemented yet for HIP)
        """
        raise NotImplementedError("Backward pass not implemented for HIP backend yet")


def causal_conv1d_hip_fn(
    x,
    weight,
    bias=None,
    activation=None,
):
    """
    HIP implementation of causal conv1d
    
    Args:
        x: (batch, dim, seqlen) - input tensor
        weight: (dim, width) - convolution weights  
        bias: (dim,) - optional bias
        activation: either None or "silu" or "swish"
    
    Returns:
        out: (batch, dim, seqlen) - output tensor
    
    Example:
        >>> batch, dim, seqlen, width = 2, 64, 512, 4
        >>> x = torch.randn(batch, dim, seqlen, device='cuda')
        >>> weight = torch.randn(dim, width, device='cuda')
        >>> bias = torch.randn(dim, device='cuda')
        >>> out = causal_conv1d_hip_fn(x, weight, bias, activation='silu')
    """
    return CausalConv1dHIPFn.apply(x, weight, bias, activation)


def causal_conv1d_hip_ref(
    x,
    weight,
    bias=None,
    activation=None,
):
    """
    Reference implementation using PyTorch operations
    For testing and validation
    
    Args:
        x: (batch, dim, seqlen)
        weight: (dim, width)
        bias: (dim,)
        activation: either None or "silu" or "swish"
    
    Returns:
        out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    
    # Use F.conv1d with padding to implement causal convolution
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    out = out[..., :seqlen]
    
    # Apply activation
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    
    return out


def test_causal_conv1d_hip():
    """
    Simple test to verify HIP implementation matches reference
    """
    print("Testing Causal Conv1D HIP implementation...")
    
    if causal_conv1d_hip_ext is None:
        print("Error: HIP extension not available. Please compile first.")
        return False
    
    # Test configuration
    batch, dim, seqlen, width = 2, 64, 512, 4
    device = 'cuda'  # In ROCm, 'cuda' maps to HIP device
    
    # Create test data
    torch.manual_seed(42)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Bias shape: {bias.shape}")
    
    # Test without activation
    print("\n[Test 1] Without activation...")
    out_hip = causal_conv1d_hip_fn(x, weight, bias, activation=None)
    out_ref = causal_conv1d_hip_ref(x, weight, bias, activation=None)
    
    diff = (out_hip - out_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Status: {'PASS ✓' if max_diff < 1e-3 else 'FAIL ✗'}")
    
    # Test with SiLU activation
    print("\n[Test 2] With SiLU activation...")
    out_hip = causal_conv1d_hip_fn(x, weight, bias, activation='silu')
    out_ref = causal_conv1d_hip_ref(x, weight, bias, activation='silu')
    
    diff = (out_hip - out_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Status: {'PASS ✓' if max_diff < 1e-3 else 'FAIL ✗'}")
    
    # Test without bias
    print("\n[Test 3] Without bias...")
    out_hip = causal_conv1d_hip_fn(x, weight, bias=None, activation='silu')
    out_ref = causal_conv1d_hip_ref(x, weight, bias=None, activation='silu')
    
    diff = (out_hip - out_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Status: {'PASS ✓' if max_diff < 1e-3 else 'FAIL ✗'}")
    
    print("\n✅ Testing complete!")
    return True


if __name__ == "__main__":
    test_causal_conv1d_hip()

