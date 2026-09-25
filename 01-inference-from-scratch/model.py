"""
Building blocks of a modern decoder-only transformer (Llama/Qwen-style), implemented
from scratch and, layer by layer, checked for correctness against Hugging Face's
reference implementation.

Blocks live here in the order we implement them:
    1. RMSNorm            (this file, first)
    2. RoPE
    3. GQA / MQA attention
    4. SwiGLU MLP
    5. Tied embeddings + full decoder block (assembles the above around the
       residual stream)

Each block is a standalone nn.Module so it can be unit-tested against a reference
before being wired into the full model.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Standard LayerNorm re-centers AND re-scales a vector:
        y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

    RMSNorm drops the re-centering (no mean subtraction, no bias) and only
    rescales by the root-mean-square of the vector:
        y = x / sqrt(mean(x^2) + eps) * weight

    Why this is enough: empirically, the re-centering step in LayerNorm
    contributes little to training stability -- the re-scaling is what matters.
    Dropping mean-subtraction removes a reduction (the mean) and the bias
    parameter, which is why RMSNorm is cheaper to compute than LayerNorm while
    reaching comparable quality. This is why every modern open decoder
    (Llama, Qwen, Mistral, ...) uses RMSNorm instead of LayerNorm.

    Where it sits in the block: RMSNorm is applied to a *copy* of the residual
    stream before attention and before the MLP ("pre-norm"). It never writes
    back into the residual stream itself -- only its *output* feeds into
    attention/MLP, whose result is then added back:

        x = x + attention(RMSNorm(x))
        x = x + mlp(RMSNorm(x))

    This matters for inference: because RMSNorm's output is thrown away after
    being consumed (it doesn't accumulate), you can freely recompute it at
    every decode step without any caching concerns -- unlike K/V, which must
    be cached.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # One learnable scale per feature dimension, initialized to 1.0 so
        # that at the start of training RMSNorm is a no-op rescale (weight=1
        # means "trust the raw normalized value").
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (..., hidden_size). We normalize over the LAST dimension
        # only -- each token's vector is normalized independently of every
        # other token and every other position in the batch.
        #
        # Do the reduction in float32 even if the model runs in bf16/fp16:
        # squaring and averaging in low precision loses too much range and is
        # a common source of subtle numerical mismatches against reference
        # implementations. This matches what Llama/Qwen's HF code does.
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # mean(x^2) over the feature dimension -> shape (..., 1)
        variance = x.pow(2).mean(dim=-1, keepdim=True)

        # rsqrt(variance + eps) is 1/sqrt(variance + eps). eps prevents a
        # divide-by-zero if a vector were ever exactly all zeros.
        x = x * torch.rsqrt(variance + self.eps)

        # Cast back to the model's working dtype, THEN apply the learned
        # per-feature scale. (HF does the multiply-by-weight in the original
        # dtype, not float32, so we match that order exactly for parity.)
        return self.weight * x.to(input_dtype)
