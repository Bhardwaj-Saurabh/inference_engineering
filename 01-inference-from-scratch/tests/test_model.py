"""
Correctness tests for model.py's building blocks.

Style used throughout this project: for each block, write a REFERENCE
computation independently (not by re-using our implementation's internals),
then assert our module matches it to floating-point tolerance. This is the
"parity test" habit the whole 12-week plan insists on -- never trust an
optimization or an implementation without a correctness check.
"""

import sys
from pathlib import Path

import torch

# Allow `import model` when running pytest from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import RMSNorm


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Independent re-derivation of RMSNorm straight from the definition, using
    plain float32 math with no fused/rsqrt tricks, as a ground truth to check
    our (more "production-style") implementation against.
    """
    x32 = x.to(torch.float32)
    mean_sq = (x32 * x32).mean(dim=-1, keepdim=True)
    denom = torch.sqrt(mean_sq + eps)
    normalized = x32 / denom
    return (weight * normalized.to(x.dtype))


def test_rmsnorm_matches_reference_formula():
    torch.manual_seed(0)
    hidden_size = 16
    norm = RMSNorm(hidden_size, eps=1e-6)
    # Give the learnable scale non-trivial values so the test would catch a
    # bug where `weight` is ignored or applied in the wrong place.
    norm.weight.data = torch.randn(hidden_size)

    x = torch.randn(4, 7, hidden_size)  # (batch, seq_len, hidden_size)

    got = norm(x)
    expected = reference_rmsnorm(x, norm.weight, norm.eps)

    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


def test_rmsnorm_preserves_shape():
    norm = RMSNorm(hidden_size=32)
    x = torch.randn(2, 5, 32)
    assert norm(x).shape == x.shape


def test_rmsnorm_unit_weight_normalizes_rms_to_one():
    """
    With weight == 1 (the init default), the RMS of the output should be ~1
    for every token, by construction: that's the entire point of the op.
    """
    norm = RMSNorm(hidden_size=64)  # weight starts at all-ones
    x = torch.randn(3, 64) * 10.0  # arbitrary scale, should be normalized away

    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()

    torch.testing.assert_close(rms, torch.ones(3), atol=1e-3, rtol=1e-3)


def test_rmsnorm_does_not_recenter():
    """
    Unlike LayerNorm, RMSNorm must NOT subtract the mean. Feeding a vector
    shifted by a large constant should change the output (a mean-subtracting
    norm would cancel the shift and leave the output unchanged).
    """
    norm = RMSNorm(hidden_size=8)
    x = torch.randn(1, 8)
    x_shifted = x + 100.0

    out = norm(x)
    out_shifted = norm(x_shifted)

    assert not torch.allclose(out, out_shifted, atol=1e-2)


if __name__ == "__main__":
    test_rmsnorm_matches_reference_formula()
    test_rmsnorm_preserves_shape()
    test_rmsnorm_unit_weight_normalizes_rms_to_one()
    test_rmsnorm_does_not_recenter()
    print("All RMSNorm tests passed.")
