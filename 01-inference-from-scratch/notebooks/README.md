# Notebooks

## `pytorch_refresher.ipynb`

A from-scratch PyTorch refresher covering exactly what's needed for this project, from
fundamentals through intermediate/advanced topics, each tied to the specific place it gets used
in `model.py` / `kv_cache.py` / `sampling.py`:

1. Tensors: creation, dtype, device
2. Shapes: `view` / `reshape` / `permute` / `transpose` / `contiguous`
3. Broadcasting
4. Indexing and slicing
5. Matrix multiplication: `@`, `matmul`, `bmm`, `einsum`
6. Autograd: `requires_grad`, `backward`, `no_grad`, `inference_mode`
7. `nn.Module`, `nn.Parameter`, buffers (walks through the `RMSNorm` already in `model.py`)
8. Numerical precision: fp32 vs bf16/fp16, and why reductions get upcast
9. Softmax and causal masking
10. The multi-head reshape pattern (split/merge heads)
11. `repeat_interleave` / `expand` for GQA's shared KV heads
12. Complex numbers for RoPE: `polar`, `view_as_complex`, `view_as_real`
13. `register_buffer` for precomputed, non-trainable state
14. Preallocated tensors + in-place writes (the KV-cache pattern), with a timed comparison
    against `torch.cat`
15. Sampling primitives: `topk`, `sort`, `cumsum`, `multinomial` (greedy/temperature/top-k/top-p)
16. Inspecting and loading real pretrained weights (`state_dict`)
17. Apple Silicon / MPS: what's actually supported on this machine (tested live, not assumed)
18. Testing numerics: `torch.testing.assert_close` and why `atol`/`rtol` matter
19. A preview of `torch.compile` and CUDA graphs (their real payoff needs a rented NVIDIA GPU)

Every code cell has been executed on this machine (Apple Silicon, MPS) — outputs in the notebook
are real, not illustrative. Where a claim about op support is version- or device-dependent (e.g.
whether MPS supports complex tensors), the notebook tests it directly rather than asserting it,
since that can change between PyTorch releases.

### Running it

```bash
uv run jupyter lab 01-inference-from-scratch/notebooks/pytorch_refresher.ipynb
```

A dedicated kernel for this project's venv is registered as **"inference-engineering (uv)"**
(installed via `uv run python -m ipykernel install --user --name inference-engineering
--display-name "inference-engineering (uv)"`). Select it if Jupyter doesn't pick it automatically.

To re-run all cells fresh from the terminal (also how this notebook's checked-in outputs were
produced):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace 01-inference-from-scratch/notebooks/pytorch_refresher.ipynb
```
