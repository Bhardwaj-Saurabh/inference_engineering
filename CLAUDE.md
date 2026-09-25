# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Progress tracking

**Read `progress.md` first, before anything else in this section.** This is a personal,
multi-session learning project worked on from more than one machine — `progress.md` is the
living status file that says exactly where things stand (current day/week in the plan, what's
implemented, what's next, environment setup already done, decisions made along the way). Update
it whenever a milestone lands: end of a work session, end of a plan "day", or any decision that
would otherwise need re-explaining to a future session.

## Project state

This is a `uv`-managed Python project for a personal 12-week learning plan on LLM inference
engineering (see `docs/plan/12-week-llm-inference-engineering-plan-v2.md`). It builds up a
portfolio of numbered sub-projects (e.g. `01-inference-from-scratch/`,
`02-gpu-kernels-and-profiling/`, `06-mini-inference-engine/`, ...) as the plan progresses — check
`docs/plan/` for the authoritative structure before assuming any layout, and see `progress.md`
for exactly how far along each one currently is.

## Commands

- Run: `uv run main.py`
- Add a dependency: `uv add <package>`
- Sync/install env: `uv sync`
- Python version is pinned via `.python-version` (3.12); managed by `uv`.

There is no test runner, linter, or formatter configured yet. When adding one, prefer `pytest` for tests and `ruff` for lint/format, and record the exact invocation here.

## The learning plan (`docs/plan/12-week-llm-inference-engineering-plan-v2.md`)

This is the single source of truth for project direction. Key structural points worth knowing before writing code in this repo:

- **Four continuous threads run across all 12 weeks**: a `perf_model.py` performance-prediction calculator (extended weekly), a from-scratch mini inference engine (the portfolio centrepiece, built in Week 6 and extended through Week 12), upstream contributions to vLLM/SGLang, and weekly interview drills.
- **Core engineering loop**: predict expected performance with `perf_model.py` *before* implementing or measuring, implement, profile, benchmark, then explicitly explain any predicted-vs-measured gap. Every benchmark table in this project should have a "Predicted" column alongside "Measured".
- **Two non-negotiable rules** baked into every week's deliverables: never claim a performance optimization without a benchmark, and never claim one without a correctness/parity check (e.g. KL divergence, top-1 agreement, or max-abs-error vs a Hugging Face reference).
- **Phases**: Weeks 1–4 (foundations: transformer internals, CUDA, Triton, quantization) → Weeks 5–6 (engine internals: reading vLLM V1/SGLang, then building a mini engine) → Weeks 7–9 (scale: multi-GPU/TP, MoE, disaggregated serving) → Weeks 10–12 (speculative decoding, correctness/determinism, capstone).
- **Spike track**: after Week 4, work commits to either Track A (kernels/performance) or Track B (distributed serving/scheduling) for the remaining weeks — check which track was chosen before assuming which later-week deliverables apply.
- **Final portfolio layout** (Section 9 of the plan) lists the expected top-level directories once the plan is underway; the mini inference engine's internal structure (`engine/`, `kernels/`, `perf_model/`, `tests/parity`, `benchmarks/`, `ci/`) is specified in that same section.

When asked to help implement a week's exercise or project, read the corresponding week's section in the plan file first — it specifies exact deliverables, benchmark table shapes, and correctness checks expected for that unit of work.

## Working style for hands-on implementation

The user's goal is to **learn the material in depth**, not just to get finished code. When
implementing plan exercises together (e.g. building `model.py` block by block):

- **One building block at a time.** Implement a single concept (e.g. RMSNorm, then RoPE, then
  GQA attention, then SwiGLU MLP, then tied embeddings) before moving to the next. Do not jump
  ahead or batch multiple blocks into one file write.
- **Detailed comments, not terse ones.** Unlike typical production code, these files should
  explain the *why* of each step inline — the math, the design reasoning, and what a bug there
  would look like — since the comments are part of the teaching material, not just the code.
- **Independent parity tests per block.** Before moving to the next block, add a test file that
  re-derives the expected behavior from the definition independently (not by re-using the
  implementation's own internals) and asserts the module matches it. This mirrors the plan's
  "never claim correctness without a check" rule, applied at the level of each small piece.
- **Pause and confirm before advancing.** After a block is implemented and its tests pass,
  check in before starting the next one rather than continuing autonomously through the whole
  file.
- Each numbered project directory (e.g. `01-inference-from-scratch/`) should end up with its own
  `README.md` explaining what's implemented, why, and how to run its tests — written as the work
  progresses, not backfilled at the end.
