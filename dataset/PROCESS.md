# Process: building and checking the problem set

The operational companion to `PLAN.md`. Each step names its script, inputs and outputs, and the check that gates the next step.
All commands run from the repository root with the venv's Python.

## 0. Environment

```bash
bash dataset/scripts/env.sh          # dataset/.venv on Python 3.12: JAX 0.10.2 CPU (root requirements.txt) + torch (requirements-port.txt)
source dataset/.venv/bin/activate
SERA_JAX_PY=$PWD/dataset/.venv/bin/python python -m unittest discover -s dataset/tests -t .
```

The oracle pins JAX 0.10.2. The system `python3` is 3.15, which has no jaxlib wheel, so `env.sh` uses Python 3.12.

**Memory.** The build machine has 17 GB of RAM, and two parallel checks on multi-GB tensors crashed it on Nov 8. Every step that
executes reference code does so in `common.Isolated`: a `python -I` subprocess in a temp directory on `JAX_PLATFORMS=cpu`, killed
above an RSS cap (5 to 8 GB per step), one at a time. Saved constructor responses are reused after an interrupted run (`--reuse-raw`).

## 1. Fetch sources: `fetch_sources.py`

Shallow clones into `dataset/raw/` (gitignored, never imported) at recorded commits: KernelBench (MIT), jax at `jax-v0.10.2`,
MaxText and tpu-inference (Apache-2.0). `raw/SOURCES.json` records repo, commit, licence sha and fetch date.

## 2. Build problems

A problem is a JAXBench-style module: `CONFIG` holds every size; `create_inputs()` takes no required arguments and is deterministic
(activations from `jax.random`, weights and norm parameters constant, exam style); `workload(*inputs)` is plain JAX and reads sizes
from `CONFIG` or array shapes.

### 2a. KernelBench ports: `port_kernelbench.py`

The constructor agent ports a KernelBench `nn.Module` with fixed semantics: Dropout is the identity, BatchNorm uses batch statistics,
conv is NCHW at the interface, `nn.Linear` weights are passed as (in, out). Shapes are scaled to 50 to 600 MB of input.

Port check: the constructor also returns a `from_torch` adapter. KernelBench's own inputs (batch dimension capped at 64 and 16M
elements per tensor, so nothing builds multi-GB tensors) go to the torch module and, through the adapter, to the JAX `workload`.
Pass: `max_rel_err <= 1e-3`. One retry with the failure text, then `excluded: port_failed`. Problems that are exam p-tasks in all
but name (matmul, RMSNorm, cross entropy, SDPA) and the 33 KernelBench stems the exam uses are skipped before any constructor call.

### 2b. Authored model operators: `author_problems.py`

The constructor writes a reference from a category spec (`constructor/prompts.py: CATEGORIES`): attention variants, norm fusions,
MoE routing, KV-cache ops, quantized matmuls, RoPE with QK-norm. Each must load, shape-trace and hold 20 MB to 9 GB of input, and
its op-chain must not equal an exam reference's (else it is sent back, at most twice).

### 2c. Public kernels: `wrap_scraped.py` then `check_scraped.py`

Only the TPU kernel folders are scanned (`jax/experimental/pallas/ops/tpu`, MaxText `kernels/`, tpu-inference `kernels/`). Files that
are exam kernels or their sources are denied: flash, paged and ragged-paged attention, Megablox, splash, the TPU matmul example,
DeepSeek V4, GDN, MHC, ragged gather/sort, SparseCore, and every file cited in `benchmarks/jaxbench/level2` headers. Paths are
de-duplicated case-insensitively (MaxText exposes `src/maxtext` and `src/MaxText`). For each survivor the constructor writes a plain-JAX
reference and a self-contained wrapper around the kernel (`<name>.kernel.py`).

**Semantics check (`check_scraped.py`).** Nothing else verifies that a scraped reference means what the kernel computes, so the two
are run side by side, one problem at a time, in an isolated subprocess on CPU:

1. Load the reference and the wrapped kernel; override `CONFIG` in both with the problem's entry in `check_scraped.SMALL`. Each
   entry keeps the kernel's block sizes and makes every dimension a multiple of them (for example the blockwise quantized matmul's
   batch 512 and in/out 1024 blocks).
2. Build inputs with the reference's `create_inputs()` for seeds 0, 1, 2 using the oracle's own seeding (`tpu_oracle.make_inputs`).
3. Run the kernel with `pallas_call` forced into interpret mode (`tpu_oracle.interpret(True)`), the reference at default precision,
   and the oracle's answer key (float32 inputs at matmul precision "highest").
4. A seed passes if `tpu_check.close(answer, kernel)` or the oracle's `xla_eq` rule holds. The problem passes if all seeds pass.
5. The result is stored as `source_ref.semantics_check` (method, small config, per-seed errors, `kernel_sha256`, upstream path,
   peak memory). A failure excludes the problem as `ref_suspect`. `--delete-kernels` then removes every `.kernel.py`; the upstream
   repo, commit and path keep the check reproducible (re-fetch, re-wrap, re-run).

Nov 8 results: spmm 1.3e-6, ragged attention 3.2e-7, blockwise quantized matmul 2.6e-3, all within tolerance. The first run caught
two harness problems, both fixed before acceptance. The ragged-attention record held an earlier, unnormalized reference version from a
duplicate wrap, and the blockwise small config did not divide the kernel's 512-row batch block.

## 3. Contamination filters: `filter_contamination.py`, `filter_neardup.py`

Against the 73 files under `benchmarks/jaxbench/` (50 level1 references, 23 level2 human kernels):

1. **KernelBench stem.** The 33 k tasks store their KernelBench origin in `CONFIG['name']`; a port with that stem (or the same name
   with another number) is out.
2. **Op-chain signature** (`signature.py`): `jax.make_jaxpr(workload)` on abstract inputs, sub-jaxprs flattened, the oracle's
   layout/dtype primitives (`outside_ops.ALLOWED`) dropped, names canonicalised, repeats collapsed. Equal to an exam reference's chain:
   out. Op-multiset Jaccard >= 0.8: listed in `review.jsonl`.
3. **Near-duplicate text.** Comments, docstrings and benchmark boilerplate stripped, literals replaced. Two token streams are scored,
   identifiers kept and identifiers abstracted (so renaming cannot hide a copy). Max 5-gram Jaccard >= 0.5 or max difflib ratio >= 0.8
   excludes. Within-set duplicates need the same signature and abstracted Jaccard >= 0.9 (GroupNorm and BatchNorm share a signature).

## 4. Oracle-readiness on CPU: `cpu_check.py`

Each reference runs alone at its full `CONFIG` size in an isolated subprocess with the oracle's files copied in, so inputs, seeds
and the answer key are built exactly as on the TPU (8 GB RSS cap, 900 s timeout). Seven checks, all required:

| Check | Meaning |
|---|---|
| `loads` | `tpu_oracle.Task(...).load()` succeeds: module imports, shapes resolve, `static_argnums` honoured |
| `fits_hbm` | 3 x input bytes <= 28e9, else the oracle never runs the task |
| `deterministic` | two `create_inputs()` calls are equal leaf by leaf |
| `seeds_distinct` | seeds 0, 1, 2 (the oracle's PRNG fold-in) give different inputs |
| `runs` | `jax.jit(workload)` executes on seed 0 |
| `finite` | every floating output is finite |
| `highest_precision` | the oracle's float32 "highest" answer key also runs; its error vs default is recorded |

Results are stored on each record as `cpu_check` and written after every problem, so a run resumes where it stopped.

## 5. Export: `export.py`

Writes `final/problems/<task_name>/baseline.py`, `final/problems.jsonl`, `final/excluded.jsonl` and `final/manifest.json`. Only
problems with status ok, a passed `cpu_check` and (scraped) a passed `semantics_check` are exported. The split is ~10 percent of
families to val by hash of `family_id`. `export.py --check` verifies the layout and every sha.

## Archive

`archive/pallas/` holds the paused Pallas pipeline (constructor kernel generation, oracle submit/collect, assembly of training
chains) with its tests. It is not run or imported by anything above.
