# Progress

Living status file for the 12-week LLM inference engineering plan
(`docs/plan/12-week-llm-inference-engineering-plan-v2.md`). Read this first when resuming work
on a different machine or in a new session — it says what's done, what's next, and any
decisions/gotchas that aren't obvious from the code alone. Update it whenever a milestone lands
(end of a work session, end of a "day", or a decision that would otherwise need re-explaining).

## Where we are

**Week 1, Day 1 — Anatomy of a modern decoder.** Working through the six building blocks
(residual stream, RMSNorm, RoPE, GQA/MQA, SwiGLU MLP, tied embeddings) as real code in
`01-inference-from-scratch/model.py`, one block at a time, before doing the hand parameter-count
exercise.

Spike track (Section 7 of the plan, chosen at end of Week 4): **not chosen yet** — too early.

## Environment

- `uv`-managed Python 3.12 project (`pyproject.toml` / `uv.lock` are committed — `uv sync` to
  reproduce).
- Dependencies added so far: `torch` (2.14, MPS confirmed working on this Apple Silicon Mac),
  `numpy`. Dev dependencies: `pytest`, `jupyter`.
- A Jupyter kernel for this venv is registered as **"inference-engineering (uv)"**. On a new
  machine, re-register it: `uv run python -m ipykernel install --user --name
  inference-engineering --display-name "inference-engineering (uv)"` (this is a local
  registration, not committed to git — must be redone per machine).
- Mac has no CUDA GPU. Per the plan, local (Mac) work is reading/napkin-math/small CPU-MPS
  experiments only; real GPU benchmarking (Week 2 onward) needs a rented NVIDIA GPU, not yet
  provisioned.

## Done so far

- Repo scaffolded, pushed to `https://github.com/Bhardwaj-Saurabh/inference_engineering` (`main`
  branch).
- `CLAUDE.md` written — repo orientation + the working-style rules for this project (one block
  at a time, detailed teaching-style comments, independent parity test per block, pause for
  confirmation before advancing, README per numbered project dir).
- `01-inference-from-scratch/data/`: two text sources downloaded (not for training — this project
  implements inference of pretrained weights, not training). `sherlock_holmes.txt` (Project
  Gutenberg, public domain) for qualitative generation checks and the Day 2 attention-parity
  test; `wikitext-2-raw/wiki.test.raw` (converted from the HF parquet mirror of
  `Salesforce/wikitext`) for the Day 27 perplexity/held-out-set exercise later. See
  `01-inference-from-scratch/data/README.md` for details and the dead-S3-mirror note.
- `01-inference-from-scratch/model.py`: **RMSNorm implemented**, with detailed inline comments
  explaining the math and design reasoning (why no mean-subtraction, why the fp32 upcast for the
  reduction, why the block is a pre-norm read-only tap on the residual stream).
- `01-inference-from-scratch/tests/test_model.py`: 4 passing parity tests for RMSNorm (matches an
  independently-written reference formula, shape preservation, RMS-of-output ≈ 1 with unit
  weight, and a check that it does NOT recenter like LayerNorm would). Run with:
  `uv run pytest 01-inference-from-scratch/tests/test_model.py -v`
- `01-inference-from-scratch/notebooks/pytorch_refresher.ipynb`: a from-scratch PyTorch refresher
  (19 sections, fundamentals through intermediate/advanced), fully executed on this machine so
  outputs are real, not hypothetical. Each section ends with a "where this shows up in the
  project" pointer. Covers: tensors/dtype/device, view/reshape/permute/contiguous, broadcasting,
  indexing, matmul/einsum, autograd/inference_mode, nn.Module/Parameter/buffers, fp32-upcast
  precision, softmax+causal masking, multi-head split/merge reshape pattern,
  repeat_interleave/expand for GQA, complex numbers for RoPE (`polar`/`view_as_complex`),
  register_buffer for precomputed constants, preallocated-tensor KV-cache pattern (with a timed
  torch.cat-vs-preallocation comparison), sampling primitives (topk/sort/cumsum/multinomial),
  loading real pretrained weights via state_dict, an Apple Silicon/MPS section, numerical-testing
  tolerances, and a torch.compile/CUDA-graphs preview. See
  `01-inference-from-scratch/notebooks/README.md`.
  - Notable finding recorded in the notebook: on this machine's torch version (2.14), MPS
    **does** support complex tensor ops (`view_as_complex`/`view_as_real`/`polar`), contradicting
    older general advice that MPS lacks complex support. The notebook tests this live rather than
    asserting it, and keeps a CPU-fallback pattern documented for portability across
    machines/versions anyway.

## Next steps (in order)

1. Implement **RoPE** in `model.py` (next building block), using notebook sections 11–13
   (complex-number rotation, register_buffer for the precomputed angle table, MPS
   live-checked instead of assumed). Add an independent parity test in `tests/test_model.py`
   before moving on.
2. Implement **GQA/MQA attention**, using the split/merge-heads reshape pattern (notebook section
   2/10) and `repeat_interleave` for shared KV heads (notebook section 11 -- and see the plan's
   own worked GQA/MQA walkthrough in the Day 1 conversation history, with the from-scratch NumPy
   `attention()` function and the group-size/`kv_head = q_head // g` mapping rule).
3. Implement **SwiGLU MLP**.
4. Assemble the full decoder block (residual stream wiring: `x = x + attn(norm(x))`, `x = x +
   mlp(norm(x))`) with tied embeddings, matching Llama-3.2-1B / Qwen3-0.6B's actual config shape.
5. Do the Day 1 hand parameter-count exercise for Qwen3-0.6B (config values still need to be
   fetched from `https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json` — not done yet)
   and verify against `sum(p.numel() for p in model.parameters())`.
6. Day 2: attention parity test against the real Hugging Face reference (needs `transformers`
   added as a dependency, not yet installed).
7. Day 3: `kv_cache.py` (contiguous KV cache, `prefill()` vs `decode_step()`), using the
   preallocated-slice-write pattern from notebook section 13.
8. Day 4: `sampling.py` (greedy/temperature/top-k/top-p/min-p), using notebook section 14's
   primitives.
9. Day 5–6: napkin math reading + build `perf_model.py`.
10. Day 7: assemble Project 1 (`01-inference-from-scratch/`) per the plan's file layout
    (`benchmark.py`, `results/`, README with the predicted-vs-measured table) and write the
    "Predicted vs measured" write-up.

## Decisions / conventions to keep consistent

- PyTorch (not NumPy) for all model code, since real HF weights and GPU work come soon.
- Every new building block: implement -> independent parity test -> confirm with user -> next
  block. Don't batch multiple blocks in one sitting.
- Data files for this project are for *testing/eval*, not training — don't add a large training
  corpus without discussing it first, the plan doesn't call for one.
- `uv add <pkg>` for anything new; keep `pyproject.toml`/`uv.lock` committed so the environment
  reproduces on another machine via `uv sync`.
