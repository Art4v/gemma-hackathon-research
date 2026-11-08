# Plan: Pallas-kernel LoRA training set in `dataset/`

Companion document: `PROCESS.md` describes the generation and collation process step by step. This file holds the decisions, the
two phases, and the verification bar.

## Context

Gemma 3n E4B is evaluated on JAXBench (65 Pallas TPU kernel tasks) using the oracle in `oracle/`: first pass plus 2 retries with
the oracle's one-line feedback. Controls are E4B bare and E4B with Pallas docs; the test arm is E4B plus a LoRA. This plan builds
the dataset that LoRA is trained on. A colleague trains it. The practice exam (held-out dev set) is someone else's job.

Scope boundary: JAXBench is the final exam only. No JAXBench task, reference, or human kernel appears in training data, and nothing
here mirrors the training recipe. The dataset must match the exam surface: the reference format, the oracle's verbatim feedback
strings, and the `workload` / `pallas_call` contract.

Terminology: the model that writes candidate kernels is the **constructor agent**, never "teacher" (that word is used elsewhere in
the study, including the oracle's optional `teacher` item field, which this pipeline does not use).

## The final oracle (Nov 8) and what it fixes in this plan

The oracle was reduced to its four steps. Facts the pipeline relies on, from `oracle/DESCRIPTION.md`, `oracle/tpu_oracle.py`,
`oracle/make_pack.py`, `oracle/QUEUE.md`:

- **Verdict chain.** `too_large_for_chip` (3x input bytes > 28 GB, not run) -> `not_a_kernel` -> `compile | vmem | mosaic | timeout |
  runtime` (`harness_oom` is the harness's fault) -> `wrong` (any of 3 seeds) -> `correct_slow` -> `pass`. `feedback` is the one line
  meant for retries. Each row carries `oracle_commit`, `oracle_sha` (hash of `tpu_oracle.py`, `tpu_check.py`, `outside_ops.py`),
  `speed_bar`, `seeds`, `jax`, `device_kind`, per-seed errors, and timing p10/p90.
- **Speed bar 0.7**, a policy choice. Reports should also give the share at parity (`speed_vs_xla >= 0.97`).
- **Seeds.** Seed 0 is `create_inputs()` as written. Seeds 1 and 2 fold the seed into every `jax.random.key` / `PRNGKey` and every
  `np.random.default_rng(int)`. Nothing else changes. Inputs not drawn from those RNGs are identical across seeds; the row records
  `seeds_distinct`. The old structure transforms (page-table permutation, redrawn weights) are gone.
- **Reference contract.** Plain JAX module with `create_inputs()` (called with no arguments; must be deterministic for a fixed seed,
  or the task fails as `harness_oom`) and `workload(*inputs)`. If the reference uses `jax.jit(workload, static_argnums=...)`, the
  solution gets the same static positions.
- **Solution contract.** Module defining `workload` with the same arguments; computation inside `pl.pallas_call`; outside it only
  layout/dtype primitives (`outside_ops.ALLOWED`) or small untainted integer metadata (<= 2^16 elements, not derived from float inputs).
- **Items.** `{id, task, ref_code, code}` or `{id, task, ref_path, code_path}`; unique ids; all items of a task share one reference.
  Optional `model` field is copied into the result row (we set it to the constructor model).
- **Queue.** Request JSON in `oracle/queue/<job_id>.json` with `job_id, owner, step, items_path|items, seeds, speed_bar, priority,
  max_chip_minutes`. `python oracle/jobq.py estimate <job_id>` must accept it. Operator runs `run.sh` on the v6e and commits
  `oracle/results/<job_id>/`. `python oracle/collect.py oracle/jobs/<job_id>` validates and writes `summary.json`.
- **Local tooling that still exists.** `tpu_oracle.py --static-only` (kernel check only, CPU, nothing run), `--cpu-smoke` (interpret
  mode, plumbing only), `estimate.task_cost(name, code, seeds)`, root `requirements.txt` (JAX 0.10.2 CPU stack).
- **Removed, so not used.** `manifest.json` and its groups, `structure.py`, `int_inputs_vary`, `keys.py`, `lower_filter` and the
  sandbox (moved to `curation/`, which this plan does not depend on).

## Decisions

| # | Decision |
|---|---|
| 1 | Contamination: exclude anything matching any of the 65 exam tasks. The exam list is the 73 `benchmarks/jaxbench/**/baseline.py` files (50 level1 references, 23 level2 human kernels). Match by KernelBench stem (`CONFIG['name']` in the 33 k baselines), by op-chain signature, and by near-duplicate text. |
| 2 | The TPU oracle is the only judge. Every kernel in the dataset has an oracle row. The only local check before submission is the oracle's own `--static-only` kernel check plus a parse check; everything else goes to the chip. |
| 3 | Example shapes: single-turn (problem -> passing kernel) and repair trajectories (problem -> failing kernel -> verbatim oracle feedback -> fixed kernel), at most 2 repair turns. Feedback turns come only from oracle rows. |
| 4 | Problem = JAXBench-style baseline module (`CONFIG`, `create_inputs()`, plain-JAX `workload`). Target = code only: one fenced Python module defining `workload` with `pl.pallas_call`. Constructor's short `plan` is metadata, not in the target. |
| 5 | Reference inputs mirror exam style: activations drawn from `jax.random` with a fixed key (so seeds 1 and 2 differ), weights and norm parameters constant (ones/zeros or deterministic), sizes read from `CONFIG`. |
| 6 | Records are structured JSONL; a reference renderer is provided but the colleague owns the chat template. |
| 7 | Sources: (a) KernelBench L1+L2 ports minus exam overlaps, with shape/dtype variants; (b) constructor-authored p-style problems in exam categories but distinct op-chains; (c) scraped public human-written Pallas TPU kernels wrapped with plain-JAX references. |
| 8 | Human-written kernels are training targets when they pass the oracle and clear the contamination filter, tagged `meta.target_origin: human`. Constructor kernels for the same problem are kept too. `--no-human-targets` disables this. |
| 9 | Constructor agent: Claude Code CLI headless (`claude -p`) under the user's Max plan for Phase 1 and the first Phase 2 batch; Anthropic Batch API (`claude-opus-5-5`) as a pluggable fallback. |
| 10 | Targets are `pass` only (>= 0.7x XLA). `stats.json` reports the parity share (>= 0.97x) alongside. `correct_slow` kernels are kept only as the failing side of repair pairs. |
| 11 | No doc-derived Q&A examples. Kernels only. |
| 12 | Every verdict carries `oracle_commit` and `oracle_sha`; `final/reverify_items.jsonl` lets a newer oracle re-judge the whole set in one job. |

Assumptions stated, not asked: scraped code permissive licenses only (Apache-2.0, MIT, BSD) with source URL and license per record;
KernelBench level 3 excluded; ~90/10 train/val split by `family_id`; `priority` 2 and `owner` = Arav on queue requests.

## Phase 1: pilot

Goal: run every stage end to end on a small set, produce a small but complete dataset, and measure what Phase 2 needs to know.

**Problem set (~35):**
- 20 KernelBench ports from the easiest families: GEMM, matmul+activation, RMSNorm/LayerNorm, softmax, elementwise chains, reductions.
- 10 constructor-authored p-style problems: 3 attention variants (sliding window, ALiBi, GQA with a head ratio not in the exam),
  2 norm+residual fusions, 2 MoE routing/top-k, 1 KV-cache update, 2 quantized matmuls.
- 5 scraped human kernels, if that many survive the filters; otherwise fewer.
- 2 shape variants of 5 of the ports, to exercise `family_id` and the split.

**Runs:** three oracle rounds (attempt 1, repair 1, repair 2), one queue job per round, `max_chip_minutes` from `jobq estimate`
plus 50 percent. Prescreen-rejected samples are also sent, flagged `probe: true`, to measure the static check's false-reject rate.

**Measurements that decide Phase 2:**
- Oracle seconds per item, by source and by verdict class (calibrates `max_chip_minutes`).
- Constructor tokens and wall time per request; the `claude -p` concurrency the Max plan tolerates without 429s.
- pass@1, pass@2, pass@3 by source. Distribution of failing verdict classes at each attempt.
- Share of `pass` kernels at parity (>= 0.97x).
- Whether conv-family KernelBench problems are worth attempting (try 3 of them in the pilot).
- Whether the Batch API fallback is needed for throughput.

**Deliverables:** `final/pilot/train.jsonl`, `val.jsonl`, `stats.json`, `MANIFEST.json`, `unsolved.jsonl`, `reverify_items.jsonl`;
`verify_dataset.py` passing; a short `PILOT_REPORT.md` with the measurements above and the Phase 2 adjustments they imply.

**Exit criteria:** every stage ran without manual patching; at least 15 accepted chains; the static check rejected no kernel that
later passed on the chip; the reverify job returns the same verdicts.

## Phase 2: the dataset

Goal: ~400 problems -> ~400 accepted chains (plus derived singles and human-kernel singles).

**Problem set:**
- All KernelBench L1+L2 ports that survive the contamination filter (~157), plus 1 to 2 shape/dtype variants each where the pilot
  pass rate for the family was above 30 percent. Conv families included only if the pilot says so.
- ~100 constructor-authored p-style problems across: attention (masking, positional, head-ratio variants), norms and residual
  fusions, MoE routing and grouped matmul at different group structure, SSM scan steps, KV-cache ops, quantized matmul and dequant
  fusions. Each checked against the exam op-chains before any kernel is written.
- All scraped human kernels that survive the filters (expected 20 to 40).

**Runs:** three oracle rounds, split into jobs of at most the per-item chip budget the pilot measured; priorities set so round r
of batch B runs before round 1 of batch B+1 when the queue is contended.

**Deliverables:** `final/train.jsonl`, `val.jsonl`, `stats.json`, `MANIFEST.json`, `unsolved.jsonl`, `reverify_items.jsonl`;
`verify_dataset.py` passing; `README.md` handoff notes for the colleague.

**Exit criteria:** >= 350 accepted chains; no exam leakage per `verify_dataset.py`; one reverify job returns identical verdicts;
distinct `oracle_sha` set has one member (or the differences are documented).

## Directory layout (`dataset/`)

```
PLAN.md  PROCESS.md  PILOT_REPORT.md  README.md
.gitignore                raw/ .venv/ candidates/raw_responses/
raw/                      UNTRUSTED clones only, never imported; read with python -I
  kernelbench/ scrape/jax/ scrape/maxtext/ scrape/tpu-inference/  SOURCES.json
problems/
  kb/<level>/<stem>[__<variant>].py   authored/<category>/<name>.py   scraped/<repo>/<name>.py (+ .kernel.py)
  problems.jsonl  excluded.jsonl  signatures.jsonl  review.jsonl
candidates/<batch>/r<round>.jsonl         constructor outputs (raw_responses/ gitignored)
prescreen/<batch>/r<round>.jsonl
submissions/<job_id>.items.jsonl + .request.json
verdicts/<job_id>.jsonl
trajectories/<batch>.jsonl
final/pilot/ final/   train.jsonl val.jsonl unsolved.jsonl stats.json MANIFEST.json reverify_items.jsonl
scripts/
  common.py  env.sh  fetch_sources.py  port_kernelbench.py  author_problems.py  wrap_scraped.py
  signature.py  filter_contamination.py  filter_neardup.py
  constructor/{__init__,prompts,claude_cli,anthropic_batch}.py
  generate.py  prescreen.py  oracle_submit.py  oracle_collect.py  assemble.py  render_chat.py  verify_dataset.py
tests/
```

Model-written and downloaded code is only ever executed in a subprocess with `python -I`, a temp cwd, and `JAX_PLATFORMS=cpu`, via the
oracle's own `--static-only` path on a pack built by `make_pack.build`. No other local execution of such code.

## Record schemas

**Problem** (`problems.jsonl`): `problem_id`, `family_id`, `task_name` (oracle `task`, `[A-Za-z0-9._-]`), `source` in {kernelbench,
authored, scraped}, `source_ref` {repo, commit, path, license, kb_level?, kb_stem?, category?, config_delta?}, `ref_code`,
`ref_sha256`, `signature` {op_chain, op_multiset, input_shapes, input_dtypes, output_shapes}, `input_bytes`, `fits_hbm`,
`est_kernel_eval_s`, `port_check|null` {method, max_rel_err, ok}, `status`, `exclude_reason|null`, `pipeline_version`, `created_at`.

**Candidate**: `candidate_id` = `f"{task_name}.a{attempt}.s{sample}"` (also the oracle item id), `problem_id`, `attempt` 1|2|3,
`sample`, `parent_candidate_id|null`, `feedback_in|null` (verbatim oracle `feedback` of the parent), `feedback_in_verdict`
{job_id, oracle_sha, verdict}|null, `prescreen_hint|null`, `constructor` {backend, model, prompt_version, docs_sha256}, `plan`,
`code`, `code_sha256`, `usage`, `created_at`.

**Prescreen**: `candidate_id`, `outcome` pass|reject|abstain, `stage` parse|static|null, `reason`, `oracle_sha_used`, `seconds`.

**Verdict**: `candidate_id` (== row.id), `job_id`, `oracle_commit`, `oracle_sha`, `verdict`, `feedback`, `speed_vs_xla|null`,
`seeds_distinct`, `accepted` (verdict == pass), `row` (full oracle row), `collected_at`.

**Training example** (`final/train.jsonl`, `val.jsonl`): `example_id`, `kind` single|repair, `derived_from|null`, `problem_id`,
`family_id`, `source`, `task_name`, `turns` = [ {user, problem, ref_code}, {assistant, kernel, code, candidate_id},
({user, oracle_feedback, feedback, verdict, verdict_ref{job_id, oracle_sha}}, {assistant, kernel, ...}) x 0..2 ], `final_verdict`,
`speed_vs_xla`, `at_parity` (>= 0.97), `n_repairs`, `oracle` {commits, shas, job_ids}, `meta` {plan, target_origin
constructor|human, constructor{backend, model, prompt_version}, failing_verdicts}, `split`.

## Verification (`verify_dataset.py`, exit 1 on any failure)

1. Schema validation of every record in `problems/`, `candidates/`, `verdicts/`, `final/`.
2. Every assistant kernel has a verdict row with matching `candidate_id`, `task_name`, `ref_sha256`, and code sha. The final turn's
   verdict is `pass`. Each non-final turn's verdict is not `pass`, and the following feedback turn equals `row.feedback` byte for byte.
3. No exam leakage: rerun the KB-stem and op-chain layers over `final/`; sha256 of every `ref_code` and `code` not among the 73
   baseline shas; near-dup scan against all 73 with the stage thresholds; report max similarity per split.
4. Provenance: distinct `(oracle_commit, oracle_sha)` set printed, warn if > 1; every row's `speed_bar` is 0.7; `reverify_items.jsonl`
   count equals distinct final kernels.
5. Split hygiene: no `family_id` in both splits; id uniqueness; `render_chat.py --check` passes.
6. `stats.json` recomputed from `train.jsonl` / `val.jsonl` and compared, including the parity share.
7. Reference hygiene: every problem's `seeds_distinct` is true in its oracle rows (activations really vary across seeds).

End to end, once per phase: submit `reverify_items.jsonl` as one oracle job and confirm every kernel is still `pass`.

Unit tests in `dataset/tests/`: op-chain canonicalisation on fake jaxprs (pattern from `oracle/tests/test_tpu_oracle.py`), assembler
on synthetic chains (pass at a1/a2/a3, `harness_oom` not consuming an attempt, prescreen re-samples excluded), feedback-verbatim
invariant, split by family, renderer round trip, contamination name match on the 33 k stems.

## Handoff notes for the colleague (go in `dataset/README.md`)

- Targets are code only; `meta.plan` is optional context, not in the default render.
- `meta.target_origin` is `constructor` or `human`. Filter on it for one style only.
- Repair examples are 3 or 5 turns; feedback turns are verbatim oracle output. If the eval harness uses the same wording, render with
  `render_chat.py` so train and eval match.
- `final/reverify_items.jsonl` re-judges the whole set under a newer oracle in one queue job.
- `final/unsolved.jsonl` lists problems no kernel passed; hard-case analysis, not training data.
- Every row records `speed_vs_xla`; `stats.json` gives both the pass rate at 0.7 and the parity share at 0.97.
