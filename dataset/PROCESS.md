# Process: generating and collating the dataset

This is the operational companion to `PLAN.md`. It describes, in order, how a problem becomes a verified training example. Run it
once for Phase 1 (pilot, ~35 problems, output under `final/pilot/`) and again for Phase 2 (~400 problems, output under `final/`).
Every step names its script, inputs, outputs, and the check that gates the next step.

## 0. Environment

```bash
bash dataset/scripts/env.sh          # venv at dataset/.venv on python3.12, installs ../requirements.txt (JAX 0.10.2 CPU)
source dataset/.venv/bin/activate
python -m unittest discover oracle/tests        # the oracle's own tests pass on this machine
python -m unittest discover dataset/tests
```

The oracle pins JAX 0.10.2 (`oracle/tpu_check.py:JAX_PIN`). The system `python3` is 3.15 and has no jaxlib wheel; `env.sh` uses
`/Users/aarav/.local/bin/python3.12`. Stage 2 additionally needs torch (`requirements-port.txt`).

## 1. Fetch sources

`python dataset/scripts/fetch_sources.py` -> `dataset/raw/`, `dataset/raw/SOURCES.json`.

Shallow clones at pinned commits: KernelBench (MIT; only `KernelBench/level1`, `level2` are used), jax at tag `jax-v0.10.2`,
MaxText, tpu-inference. `SOURCES.json` records repo, commit, license file sha, fetch date. Nothing under `raw/` is ever imported or
executed by the pipeline; files are read as text.

## 2. Build problems

A problem is a plain-JAX baseline module in JAXBench style:

```python
CONFIG = {"name": "<stem>", "batch": 4096, "features": 8192, ...}     # every size lives here
def create_inputs():                                                 # no arguments; deterministic for a fixed seed
    key = jax.random.key(0)                                          # activations: drawn from jax.random so seeds 1 and 2 differ
    x = jax.random.normal(key, (CONFIG["batch"], CONFIG["features"]), jnp.bfloat16)
    w = jnp.ones((CONFIG["features"],), jnp.float32)                 # weights/norm params: constant, exam style
    return x, w
def workload(x, w):                                                  # sizes from CONFIG or array shapes, never literals
    ...
```

Rules every problem must satisfy before it is recorded: 3x input bytes <= 28e9 (else the oracle never runs it); `create_inputs()`
returns identical values on two calls; `jax.eval_shape(workload)` succeeds on CPU; dims multiples of 128 where the op allows (8 on
sublane dims); if `jax.jit(workload, static_argnums=...)` is used, the static positions are Python ints.

### 2a. KernelBench ports: `port_kernelbench.py`

Input: `raw/kernelbench/KernelBench/level{1,2}/*.py` (a PyTorch `Model`, `get_inputs`, `get_init_inputs`).
Output: `problems/kb/<level>/<stem>.py`, problem records.

For each file the constructor agent is asked to port the module under the rules above, with these semantics fixed: Dropout is
identity (inference); BatchNorm uses batch statistics (JAXBench 31k convention); conv is NCHW at the interface, NHWC inside, like
18k; `nn.Linear` weight is `(out, in)` and the port must transpose accordingly. Shapes are scaled up from KernelBench's small
defaults to 50 to 600 MB of input.

Port check (isolated subprocess, `python -I`): at KernelBench's own shapes with the batch dimension capped at 64, build one set of
inputs and weights, feed the same values to the torch `Model` (batch-statistics BatchNorm, Dropout off) and to the JAX `workload`
through the constructor's `from_torch` adapter, require `max_rel_err <= 1e-3`. One retry with the diff, then `excluded: port_failed`.

**Memory.** The build machine has 17 GB of RAM. KernelBench's "small" inputs reach 6 GB per tensor, and two parallel checks crashed the
machine on Nov 8. Checks therefore run one at a time (`--check-workers 1`), every isolated subprocess is killed above 5 GB resident
(`common.Isolated(max_rss_gb)`), and the batch dimension is shrunk before any tensor is built. Saved constructor responses in
`candidates/raw_responses/` are reused with `--reuse-raw` after an interrupted run, so a crash costs no constructor calls.

Then 1 to 2 shape/dtype variants per port (same `family_id`, `config_delta` recorded) once Phase 1 shows the family is solvable.

### 2b. Constructor-authored p-style problems: `author_problems.py`

Input: a category list in `scripts/constructor/prompts.py` (attention variants, norm+residual fusions, MoE routing and grouped
matmul, SSM scan steps, KV-cache ops, quantized matmul). Output: `problems/authored/<category>/<name>.py`.

The constructor writes a baseline from a one-paragraph spec per category and the rules above. The script immediately runs the
contamination filter (step 3) on the new problem; a match sends the spec back with the matching exam task named and asks for a
different operator, at most twice. Solvability is proven later by a passing kernel in step 5; problems with none after three
attempts end in `unsolved.jsonl`.

### 2c. Scraped human kernels: `wrap_scraped.py`

Input: files under `raw/scrape/**` containing `pl.pallas_call`. Output: `problems/scraped/<repo>/<name>.py` (reference) and
`<name>.kernel.py` (the human kernel, rewrapped).

Denylist first: every upstream file that is an exam human kernel (flash_attention, paged_attention, megablox/gmm, splash,
ragged_paged_attention, the TPU matmul example, and every source path cited in `benchmarks/jaxbench/level2/*/baseline.py` headers).
For each survivor the constructor writes (a) a plain-JAX reference with `create_inputs()` at realistic sizes and (b) a self-contained
`workload` module that inlines the human kernel, strips unrelated config paths, and keeps only layout/dtype ops outside
`pallas_call`. The human kernel is submitted to the oracle as item `<task>.human.s0` in round 1. If it is `wrong`, the reference is
suspect and the problem is excluded (`ref_suspect`). If it passes, it becomes a single-turn example with `target_origin: human`.

## 3. Contamination and duplicate filters

`python dataset/scripts/filter_contamination.py && python dataset/scripts/filter_neardup.py`
-> `problems/problems.jsonl` (kept), `problems/excluded.jsonl` (with reason and the matching exam task), `problems/review.jsonl`
(near matches for a human), `problems/signatures.jsonl` (audit trail).

The exam set is the 73 files under `benchmarks/jaxbench/`: 50 `level1/*/baseline.py` references and 23 `level2/*/baseline.py`
human kernels. There is no manifest; task names are the directory names. Three layers, any hit excludes:

1. **KernelBench stem.** The 33 k-task baselines store their KernelBench origin in `CONFIG['name']` (for example
   `41k_Gemm_Add_ReLU` holds `'76_Gemm_Add_ReLU'`). A port whose stem equals any of those 33 is out.
2. **Op-chain signature** (`signature.py`). `jax.make_jaxpr(workload)` on abstract inputs, sub-jaxprs flattened with
   `oracle/outside_ops._sub_jaxprs`, layout/dtype primitives in `outside_ops.ALLOWED` dropped, names canonicalised, consecutive
   repeats collapsed. Equal chain to any of the 50 level1 references -> excluded regardless of shape. Op-multiset Jaccard >= 0.8 ->
   `review.jsonl`.
3. **Near-duplicate text.** Comments and docstrings stripped, literals replaced by a placeholder, then 5-gram token Jaccard and
   `difflib.SequenceMatcher.ratio()` against all 73 files, for both the reference and (for scraped) the kernel. Two token streams are
   scored, identifiers kept and identifiers abstracted (so renaming variables cannot hide a copy); the max Jaccard >= 0.5 or the max
   ratio >= 0.8 excludes. A ratio in the 0.6s is what two unrelated modules sharing the JAXBench skeleton score, so it does not
   exclude. Top-3 nearest recorded for every kept problem.

Within-set dedupe: identical signature and shapes -> keep one.

Nothing is sent to the constructor for kernel writing until the problem is in `problems.jsonl`.

## 4. Constructor generation: `generate.py --batch B --round r`

Input: `problems.jsonl`, `trajectories/B.jsonl`, `verdicts/*.jsonl`. Output: `candidates/B/r<r>.jsonl`.

The constructor sees exactly what the student will see: `render_chat.py` renders the user turn (fixed instruction + fenced
reference) and, on repair rounds, the prior kernel and `"Oracle verdict: " + feedback` verbatim. The system prompt carries the
oracle's rules (the `not_a_kernel` wording, the outside-ops rule, v6e, JAX 0.10.2, `workload` with `pl.pallas_call`, sizes from
shapes, no `interpret=True`, target >= 0.7x XLA) and `pallas_guides/teacher_docs.md` (sha recorded). Output is structured
`{plan, code}`; `plan` goes to metadata, `code` is the candidate.

Backends behind one `Constructor` protocol: `claude_cli.py` runs
`claude -p --output-format json --json-schema ... --system-prompt-file ... --no-session-persistence --allowedTools ""` in a
thread pool; `anthropic_batch.py` sends the same requests through Message Batches. Raw responses are kept under
`candidates/raw_responses/` (gitignored); records hold only the extracted fields plus token usage.

Rounds:
- **Round 1:** attempt 1 for every problem.
- **Round 2:** for each candidate whose round-1 verdict is kernel-attributable and not `pass` (`wrong, not_a_kernel, compile, vmem,
  mosaic, runtime, timeout, correct_slow`): attempt 2 with the verbatim feedback.
- **Round 3:** same, with the full history. After three attempts the problem is `unsolved`.
- `harness_oom` and `too_large_for_chip` are the harness's fault: no attempt consumed, never shown as feedback, problem marked
  `harness_blocked` (shrink the CONFIG as a new variant, or drop).

## 5. Local pre-check: `prescreen.py`

Input: a candidates file. Output: `prescreen/B/r<r>.jsonl`. Reject-only; the oracle is the judge.

1. **parse:** one Python module, `ast.parse` succeeds, defines `workload`, imports `jax.experimental.pallas`.
2. **static:** the oracle's own kernel check. Build a throwaway pack with `oracle/make_pack.build` and run
   `python tpu_oracle.py pack.json out.jsonl --static-only` in a subprocess (`python -I`, temp cwd, `JAX_PLATFORMS=cpu`). `not_a_kernel`
   -> reject with the oracle's reason; `static_ok` -> pass; any harness error -> abstain.

A rejected sample never reaches the oracle and never appears in a trajectory. Its slot is re-sampled with the reason as a private
hint (`prescreen_hint`), at most twice; if every re-sample is rejected the last one is submitted anyway so the chain gets a real
verdict. In Phase 1, rejected samples are also submitted flagged `probe: true` to measure false rejections.

## 6. Oracle submission and collection

`python dataset/scripts/oracle_submit.py --batch B --round r`
-> `oracle/queue/dataset-B-r<r>.json` + `dataset-B-r<r>.items.jsonl`, copies in `dataset/submissions/`.

Items are `{id: candidate_id, task: task_name, ref_code, code, model}`; items of one task share the problem's `ref_code` by
construction. The request sets `seeds: 3`, `speed_bar: 0.7`, `priority: 2`, `owner`, `step: dataset-r<r>`, and `max_chip_minutes`
from `python oracle/jobq.py estimate dataset-B-r<r>` plus 50 percent. If the estimate is refused, the script splits the job by task
and retries. Commit both files and tell the operator.

The operator builds (`jobq.py build`), runs `run.sh` on the v6e as the locked-down user (`oracle/QUEUE.md`), and commits
`oracle/results/dataset-B-r<r>/`.

`python dataset/scripts/oracle_collect.py dataset-B-r<r>` -> `verdicts/dataset-B-r<r>.jsonl`.
Checks before accepting: `oracle/collect.py` reports `valid: true`; one row per submitted id; a single `oracle_sha`;
`seeds == [0,1,2]`; `speed_bar == 0.7`; every row's `ref_sha256` matches the problem's; `seeds_distinct` is true (else the problem's
`create_inputs` does not vary across seeds and the problem is flagged). Each verdict is appended to `trajectories/B.jsonl`.

Steps 4 to 6 repeat for rounds 1, 2, 3.

## 7. Assembly: `assemble.py --phase pilot|full`

Input: `trajectories/*.jsonl`, `verdicts/*.jsonl`, `problems.jsonl`. Output: `final/[pilot/]`.

Emission rules, per problem chain:
- a1 `pass` -> `single` [P, a1].
- a2 `pass` -> `repair` [P, a1, fb1, a2] and `single` [P, a2] with `derived_from`.
- a3 `pass` -> `repair` [P, a1, fb1, a2, fb2, a3] and `single` [P, a3]. The suffix [P, a2, fb2, a3] is emitted only with `--emit-suffix`.
- Human kernel `pass` -> `single` [P, human] with `target_origin: human`.
- No `pass` in three attempts -> `unsolved.jsonl`.
- `correct_slow` never ends a chain; it appears only as a failing turn inside a repair example.

Split: ~10 percent of `family_id`s to `val`, chosen by hash of the family id so it is stable; all variants of a family land together.

`stats.json`: counts by source, kind, `n_repairs`, `target_origin`; failing-verdict classes repaired at each attempt; `speed_vs_xla`
quantiles; parity share (`speed_vs_xla >= 0.97`); code-length quantiles; pass@1/2/3 per source and per family; unsolved list.
`MANIFEST.json`: sha256 of every final file, the set of `oracle_commit`/`oracle_sha`, prompt versions, docs sha, pipeline version.
`reverify_items.jsonl`: every final kernel as `{id, task, ref_code, code}`.

## 8. Verification: `verify_dataset.py final/[pilot/]`

Runs the checks in `PLAN.md` "Verification" and exits 1 on any failure. Then submit `reverify_items.jsonl` as one queue job
(`dataset-reverify-<phase>`) and confirm with `oracle_collect.py --expect-pass` that every kernel is still `pass`.

## 9. Render check and handoff

`python dataset/scripts/render_chat.py --check final/[pilot/]` round-trips every example through the reference chat renderer and
asserts the fenced code equals `code` byte for byte. The renderer is offered to the eval-harness owner so train and eval formats
can match. `README.md` carries the handoff notes listed in `PLAN.md`.

## Phase 1 specifics

Problem set: 20 easy KernelBench ports (+2 variants on 5 of them), 10 authored p-style, up to 5 scraped, 3 conv ports as a probe.
Output under `final/pilot/`. Write `PILOT_REPORT.md` with: oracle seconds per item by source and verdict; constructor tokens and
wall time per request and the sustainable `claude -p` concurrency; prescreen reject and false-reject rates; pass@1/2/3 by source;
parity share; conv verdict; whether the Batch API fallback is needed. Phase 2 proceeds only when the Phase 1 exit criteria in
`PLAN.md` are met.

## Phase 2 specifics

Problem set: all surviving KernelBench ports with variants where the pilot pass rate for the family exceeded 30 percent, ~100
authored p-style problems, all surviving scraped kernels. Jobs sized to the pilot's measured per-item chip time. Output under `final/`.

## Invariants (checked by `verify_dataset.py`)

- Every kernel in `final/` has an oracle row that says `pass`, with matching ids, task, reference sha and code sha.
- Every feedback turn is the oracle's `feedback` for the kernel in the preceding turn, byte for byte.
- No JAXBench reference, kernel, KernelBench stem used by the exam, or exam op-chain appears in `final/`.
- No `family_id` appears in both splits.
- Every row was judged with `speed_bar` 0.7 and `seeds` [0,1,2], by a recorded `oracle_sha`.
