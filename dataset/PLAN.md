# Plan: Pallas-kernel LoRA training set in `dataset/`

## Context

Gemma 3n E4B is evaluated on JAXBench (65 Pallas TPU kernel tasks, scored by `oracle/tpu_oracle.py`): first pass plus 2 retries with the oracle's one-line feedback. Controls are E4B bare and E4B with Pallas docs; the test arm is E4B plus a LoRA. This plan builds the dataset that LoRA is trained on. A colleague trains it; the practice exam (held-out dev set) is someone else's job.

Scope boundary set by the user: JAXBench is the final exam only. No JAXBench task, reference, or human kernel appears in training data, and nothing here mirrors the training recipe. The dataset must match the exam surface: the reference format, the oracle's verbatim feedback strings, and the `workload` / `pallas_call` contract.

**Action for the colleague who owns the oracle:** `oracle/manifest.json` still marks 21 tasks (3p-6p, 29k, 45k, 51p-65p) as `practice_problem` with `train_allowed: true`, and `benchmarks/jaxbench/README.md` describes that group as cleared for training. All 65 should become `final_exam` with `train_allowed` and `teacher_allowed` false. This plan already treats all 65 as exam material regardless.

Terminology: the model that writes candidate kernels is the **constructor agent** (never "teacher"; that word is used elsewhere in the study).

## Decisions (settled with the user)

| # | Decision |
|---|---|
| 1 | Contamination: exclude anything matching any of the 65 exam tasks (name, op-chain signature, near-dup against all 73 `benchmarks/jaxbench/**/baseline.py`). |
| 2 | Every kernel in the dataset is verified by the TPU oracle (single source of truth). CPU interpret-mode is only a pre-screen to save queue time. |
| 3 | Example shapes: single-turn (problem -> passing kernel) and repair trajectories (problem -> failing kernel -> verbatim oracle feedback -> fixed kernel), max 2 repair turns. Repair feedback comes only from TPU verdicts. |
| 4 | Problem = JAXBench-style baseline module (`CONFIG`, `create_inputs(dtype)`, plain-JAX `workload`). Target = code only: one fenced Python module defining `workload` with `pl.pallas_call`. Constructor's short `plan` stored as metadata, not in the target. |
| 5 | Records are structured JSONL; a reference renderer is provided but the colleague owns the chat template. |
| 6 | Size: Small tier, ~400 problems -> ~400 accepted examples. |
| 7 | Sources: (a) KernelBench L1+L2 ports minus exam overlaps, with shape/dtype variants; (b) constructor-authored p-style problems (~100) in exam categories but distinct op-chains; (c) scraped public human-written Pallas TPU kernels (~20-40 after filtering) wrapped with plain-JAX references. |
| 7b | Human-written kernels ARE training targets when they pass the oracle and clear the contamination filter, tagged `meta.target_origin: human`. Constructor kernels for the same problem are also kept, so one problem can yield two single-turn examples with different styles. `--no-human-targets` disables this. |
| 8 | Constructor agent: Claude Code CLI headless (`claude -p`) under the user's Max plan for the 30-problem pilot and first batch; Anthropic Batch API (`claude-opus-5-5`) as a pluggable fallback. |
| 9 | No doc-derived Q&A examples. Kernels only. |
| 10 | Verification is pluggable; every verdict carries oracle commit/sha. Speed-bar question (`pass` only vs also `correct_slow`) deferred: default `pass` only, `--accept-correct-slow` flag. |

Assumptions stated, not asked: scraped code permissive licenses only (Apache-2.0, MIT, BSD) with source URL and license per record; KernelBench level 3 excluded; ~90/10 train/val split by `family_id`.

## Repo facts the design relies on

- Oracle verdict chain (`oracle/tpu_oracle.py`): `not_a_kernel` -> `compile|vmem|mosaic|timeout|runtime|harness_oom` -> `wrong` (3 seeds, rtol/atol 1e-2 or xla_eq) -> `correct_slow` (speed_vs_xla < 0.8) -> `pass`. `feedback` is the one line designed for retries. Hardware v6e-1, JAX 0.10.2.
- Queue contract (`oracle/QUEUE.md`, `oracle/make_pack.py:load_items`): items `{id, task, ref_code, code}`, unique ids, all items of a task share one `ref_code`.
- Exam k-tasks are all KernelBench **level 2**; numbering is not 1:1 (`41k_Gemm_Add_ReLU` is KB L2 #76) but each k baseline stores the KB stem in `CONFIG['name']`.
- `pallas_guides/teacher_docs.md` exists (docs-control material); the constructor's system prompt can reuse it.
- Local: `python3` is 3.15 (no jaxlib wheel); use `/Users/aarav/.local/bin/python3.12`. JAX not installed. `claude` CLI 2.1.319 present with `-p`, `--output-format json`, `--json-schema`, `--system-prompt-file`.
- `--cpu-smoke` in the oracle runs at full `create_inputs` size; the pre-screen needs a shrunk CONFIG copy (`ref_code_small`), pattern in `oracle/tests/test_tpu_oracle.py` (`_shrunk()`, `DRIVER`).

## Directory layout (`dataset/`)

```
README.md                 stages, env setup, invariants, colleague handoff notes
requirements.txt          jax==0.10.2 jaxlib==0.10.2 numpy>=2 ml_dtypes>=0.5 ; requirements-port.txt adds torch
.gitignore                raw/ .venv/ candidates/raw_responses/
raw/                      UNTRUSTED clones only, never imported; read with python -I
  kernelbench/ scrape/jax/ scrape/maxtext/ scrape/tpu-inference/  SOURCES.json (repo, commit, license, date)
problems/                 baseline modules = user-turn material
  kb/<level>/<stem>[__<variant>].py     ported KernelBench baselines (+ .small.py shrunk copy)
  authored/<category>/<name>.py         constructor-authored p-style baselines
  scraped/<repo>/<name>.py (+ .kernel.py) plain-JAX reference wrapping a scraped kernel
  problems.jsonl  excluded.jsonl  signatures.jsonl
candidates/<batch>/r<round>.jsonl   constructor outputs (raw_responses/ gitignored)
prescreen/<batch>/r<round>.jsonl
submissions/<job_id>.items.jsonl + .request.json   copies of what went to oracle/queue/
verdicts/<job_id>.jsonl             ingested from oracle/results/<job_id>/
trajectories/<batch>.jsonl          per-problem attempt chains
final/  train.jsonl val.jsonl unsolved.jsonl stats.json MANIFEST.json reverify_items.jsonl
scripts/
  common.py  env.sh  fetch_sources.py  port_kernelbench.py  author_problems.py  wrap_scraped.py
  signature.py  filter_contamination.py  filter_neardup.py
  constructor/{__init__,prompts,claude_cli,anthropic_batch}.py
  generate.py  prescreen.py  oracle_submit.py  oracle_collect.py  assemble.py  render_chat.py  verify_dataset.py
tests/                    unittest, oracle/tests style; JAX tests skipped unless SERA_JAX_PY set
```

Any execution of downloaded or model-written code runs under `sandbox-exec` with `oracle/sandbox/sandbox_profile.py:_profile`, `python -I`, cwd = temp workdir holding a `make_pack.build()` pack (profile denies `$HOME` reads).

## Record schemas

**Problem** (`problems.jsonl`): `problem_id`, `family_id` (variants share it; drives the split), `task_name` (oracle `task`, `[A-Za-z0-9._-]`), `source` in {kernelbench, authored, scraped}, `source_ref` {repo, commit, path, license, kb_level?, kb_stem?, category?, config_delta?}, `ref_code`, `ref_sha256`, `ref_code_small|null`, `signature` {op_chain, op_multiset, input_shapes, input_dtypes, output_shapes}, `input_bytes`, `fits_hbm`, `est_kernel_eval_s` (from `oracle/estimate.py:task_cost`), `port_check|null` {method, max_rel_err, ok}, `status`, `exclude_reason|null`, `pipeline_version`, `created_at`.

**Candidate**: `candidate_id` = `f"{task_name}.a{attempt}.s{sample}"` (also the oracle item id), `problem_id`, `attempt` 1|2|3, `sample`, `parent_candidate_id|null`, `feedback_in|null` (VERBATIM oracle feedback of parent), `feedback_in_verdict` {job_id, oracle_sha, verdict}|null, `prescreen_hint|null` (never a training turn), `constructor` {backend, model, prompt_version, docs_sha256}, `plan`, `code`, `code_sha256`, `usage`, `created_at`.

**Prescreen**: `candidate_id`, `outcome` pass|reject|abstain, `stage` parse|static|lower|interpret|null, `reason`, `prescreen_version`, `oracle_sha_used`, `seconds`.

**Verdict**: `candidate_id` (== row.id), `job_id`, `oracle_commit`, `oracle_sha`, `verdict`, `feedback`, `speed_vs_xla|null`, `accepted` (verdict == pass, or correct_slow with flag), `row` (full oracle row, verbatim), `collected_at`.

**Training example** (`final/train.jsonl`, `val.jsonl`): `example_id`, `kind` single|repair, `derived_from|null`, `problem_id`, `family_id`, `source`, `task_name`, `turns` = [ {user, problem, ref_code}, {assistant, kernel, code, candidate_id}, ({user, oracle_feedback, feedback, verdict, verdict_ref{job_id, oracle_sha}}, {assistant, kernel, ...})×0..2 ], `final_verdict`, `speed_vs_xla`, `n_repairs`, `oracle` {commits, shas, job_ids}, `meta` {plan, constructor{backend, model, prompt_version}, failing_verdicts}, `split`.

## Pipeline stages

0. **`env.sh`**: venv on python3.12, install requirements, export `SERA_JAX_PY`; run `python -m unittest discover oracle/tests` once to confirm the oracle CPU path works locally.
1. **`fetch_sources.py`** -> `raw/`, `SOURCES.json`. Shallow clones at pinned commits: KernelBench (MIT), jax at `jax-v0.10.2` (commit from `pallas_guides/SOURCES.json`), MaxText, tpu-inference. Nothing executed.
2. **`port_kernelbench.py`** (KB L1+L2 -> `problems/kb/`). Constructor converts each `Model` to the baseline shape under hard rules: all sizes in `CONFIG` with `'name': '<kb_stem>'`; deterministic `create_inputs(dtype=jnp.float32)` returning inputs and weights (norm affine as ones/zeros, like exam k baselines); `workload` reads sizes from CONFIG or array shapes, never literals; inference semantics for Dropout, batch statistics for BatchNorm, NCHW<->NHWC like 18k; shapes scaled to 50-600 MB input, dims multiples of 128 where possible. **Torch equivalence check** in the sandbox at KB's own small shapes, same weights, `max_rel_err <= 1e-3`; one constructor retry with the diff, else `excluded: port_failed`. Writes `ref_code_small`, `signature`, cost fields. Then emit 1-2 shape/dtype variants per surviving port (same `family_id`) to reach the count. Order: gemm/norm/activation/reduction families first, conv families last (lowest expected pass rate).
3. **`author_problems.py`** (-> `problems/authored/`). Constructor writes ~100 new p-style baselines from a category list (attention variants with different masking/positional schemes, norm+residual fusions, MoE routing/top-k, SSM scan steps, KV-cache ops, quantized matmul) at realistic bf16 shapes. Each must: trace and `eval_shape` on CPU, run at small size, pass `int_inputs_vary` where it has int inputs, and clear the contamination filter (stage 5). Authored references are validated for solvability by the constructor's own passing kernel in stage 6; a problem with no passing kernel after 3 attempts lands in `unsolved.jsonl`.
4. **`wrap_scraped.py`** (`raw/scrape/**` -> `problems/scraped/`). Enumerate files containing `pl.pallas_call`; denylist upstream files that are exam human kernels (flash_attention, paged_attention, megablox/gmm, splash, ragged_paged_attention, matmul.py, and every source path cited in `level2/*/baseline.py` headers). For survivors the constructor writes a plain-JAX `ref_code` plus a `workload` wrapper around the scraped kernel. The scraped kernel is submitted alongside constructor attempts (`<task>.human.s0`). It serves two purposes: a reference-validity probe (`wrong` -> problem excluded as `ref_suspect`, since a production kernel disagreeing with our reference means the reference is wrong) and, when it passes, a training target in its own right with `meta.target_origin: human`. Human kernels only produce single-turn examples (there is no human repair trajectory). The wrapper around the kernel must keep the module self-contained: inline the kernel source, strip unrelated config paths, keep only layout/dtype ops outside `pallas_call`.
5. **`filter_contamination.py`** + **`filter_neardup.py`**, against all 65 tasks:
   - Name: KB stem == `CONFIG['name']` of any exam k baseline (33 names).
   - Op-chain signature (`signature.py`): `jax.make_jaxpr(workload)` on abstract shapes in the sandbox; flatten sub-jaxprs via `oracle/outside_ops.py:_sub_jaxprs`; drop `outside_ops.ALLOWED` layout/dtype prims; canonicalise and collapse repeats; equal chain to any exam task -> excluded regardless of shape. Op-multiset Jaccard >= 0.8 -> `review.jsonl` for a human.
   - Near-dup: normalised-token 5-gram Jaccard and `difflib` ratio of every `ref_code` and kernel against all 73 baseline files; drop at Jaccard >= 0.5 or ratio >= 0.6; record top-3 nearest per kept problem in `signatures.jsonl`.
   - Within-set dedupe on identical signature + shapes.
6. **`generate.py --batch B --round r`**: the attempt state machine (below). Constructor protocol `generate(list[Request]) -> list[Response]`; `claude_cli.py` = `claude -p --output-format json --json-schema {plan, code} --system-prompt-file ... --no-session-persistence --allowedTools ""` with a thread pool; `anthropic_batch.py` = Message Batches with the same requests (load the `claude-api` skill at implementation time for exact SDK shapes). System prompt = rules distilled from the oracle (quote the `not_a_kernel` text and the outside-ops rule, v6e, JAX 0.10.2, module must define `workload` with `pl.pallas_call`, sizes from shapes, no `interpret=True`, >= 0.8x XLA) + `pallas_guides/teacher_docs.md` (sha recorded). User turn rendered by `render_chat.py` so the constructor sees what the student will see.
7. **`prescreen.py`**: reject-only, all in sandbox. (a) parse: one module, defines `workload`, imports pallas; (b) static `not_a_kernel` via packed `tpu_oracle.py --static-only` (reuses `kernel_check`, `tpu_oracle.py:151`); (c) lowering via `oracle/lower_filter.py:lower_sandboxed` + `outcome`; (d) interpret correctness, 3 seeds, on `ref_code_small` via the `DRIVER` pattern (`o.init(cpu_smoke=True)`, timer mocked). `wrong` on all seeds -> reject; exception in interpret -> abstain. Cannot catch vmem/mosaic/runtime/timeout/speed; those go to the chip.
8. **`oracle_submit.py`** writes `{id: candidate_id, task, ref_code, code}` items + request JSON per `QUEUE.md` (`job_id: dataset-B-r<r>`, seeds 3, speed_bar 0.8, `max_chip_minutes`) into `oracle/queue/`, copies to `submissions/`, runs `python oracle/jobq.py estimate` and splits by task if over budget. **`oracle_collect.py`** ingests `oracle/results/<job_id>/`, checks one row per id, single `oracle_sha`, `seeds == [0,1,2]`, `ref_sha256` matches, writes `verdicts/`. Reuse `oracle/collect.py:check` where a MANIFEST exists.
9. **`assemble.py`** -> `final/`: examples from trajectories whose last verdict is accepted; split by `family_id`; `stats.json` (counts by source/kind/n_repairs, verdict classes repaired, speed quantiles, pass@1/2/3 per source); `MANIFEST.json` (sha256 of every file, oracle sha/commit set, prompt versions, docs sha); `unsolved.jsonl`; `reverify_items.jsonl` (every final kernel as an oracle item, so a newer oracle can re-judge the whole set in one job).
10. **`render_chat.py`**: `render(example) -> list[{role, content}]`. User = fixed short instruction + fenced `ref_code`; assistant = fenced `code` only; feedback turn = `"Oracle verdict: <feedback>"` + fixed retry instruction. `--check` round-trips every example and asserts the fenced code equals `code` byte for byte. Offered to the eval-harness owner so train and eval formats can match.

## Attempt state machine (faithful repair trajectories)

Rounds, because the oracle is an operator-run batch queue.
- Round 1: attempt 1 for every problem -> prescreen -> oracle job r1.
- Round 2: for candidates with a kernel-attributable non-accepted verdict (`wrong, not_a_kernel, compile, vmem, mosaic, runtime, timeout, correct_slow`): attempt 2 prompt = problem + attempt-1 code + `"Oracle verdict: " + row.feedback` verbatim -> prescreen -> job r2.
- Round 3: same with full history -> job r3. Max 3 attempts; then `unsolved`.
- `harness_oom` / `too_large_for_chip` are not the kernel's fault: no attempt consumed, never shown as feedback, problem marked `harness_blocked` (shrink as a new variant or drop).
- Prescreen-rejected samples never reach the oracle or any trajectory. The slot is re-sampled (new `sample`, hint recorded as `prescreen_hint`), at most 2 re-samples; if all rejected, the last goes to the oracle anyway so the chain has a real verdict. `abstain` -> straight to oracle.
- Invariant: every `oracle_feedback` turn is the oracle's `feedback` for exactly the kernel in the preceding assistant turn (`candidate_id == row.id`, code sha matches).
- Emission: a1 pass -> `single`. a2 pass -> `repair` [P,a1,fb1,a2] + `single` [P,a2] with `derived_from`. a3 pass -> `repair` [P,a1,fb1,a2,fb2,a3] + `single` [P,a3]; suffix [P,a2,fb2,a3] behind `--emit-suffix` (default off). The ~400 target counts chains, not derived singles.

## Pilot (before any large batch)

30 problems: 20 easy KernelBench ports (gemm/norm/activation) + 10 constructor-authored p-style. Run stages 2/3 -> 5 -> 6/7 -> 8, three rounds. Measure: constructor tokens and wall time per request, `claude -p` concurrency the Max plan tolerates, prescreen reject rate by stage, oracle seconds per item (calibrates `max_chip_minutes`), verdict distribution, pass@1/2/3, prescreen false-reject rate (send rejected samples flagged `probe: true`). Decide from it: per-item chip budget, whether `correct_slow` is common enough to keep as a tier, how many conv problems to attempt, whether the Batch API fallback is needed.

## Sequencing

1. `env.sh`, `common.py`, tests scaffold. 2. `fetch_sources.py`. 3. `signature.py` + both filters (nothing leaves the machine before this works). 4. Constructor protocol, prompts, `claude_cli.py`, `render_chat.py`. 5. `port_kernelbench.py` on ~20 easy problems + `author_problems.py` on 10. 6. `prescreen.py`. 7. `generate.py`, `oracle_submit.py`, `oracle_collect.py`. 8. **Pilot.** 9. Remaining ports + variants, `author_problems.py` to ~100, `wrap_scraped.py`. 10. Full batches, `assemble.py`, `verify_dataset.py`, README handoff notes.

## Verification (`verify_dataset.py`, exit 1 on any failure)

1. Schema validation of every record in `problems/`, `candidates/`, `verdicts/`, `final/`.
2. Every assistant kernel has a verdict row with matching `candidate_id`, `task_name`, `ref_sha256`, and code sha; final turn accepted; non-final turns non-accepted with the next feedback turn equal to `row.feedback` byte for byte.
3. No exam leakage: rerun name + op-chain layers over `final/`; sha256 of every `ref_code` and `code` not in the set of all 73 baseline shas; near-dup scan against all 73 with stage-5 thresholds; report max similarity per split.
4. Provenance: distinct `(oracle_commit, oracle_sha)` set printed, warn if > 1; `reverify_items.jsonl` count equals distinct final kernels.
5. Split hygiene: no `family_id` in both splits; id uniqueness; `render_chat.py --check` passes.
6. `stats.json` recomputed from `train.jsonl`/`val.jsonl` and compared.
End-to-end: after the pilot, submit `reverify_items.jsonl` as one oracle job and confirm every kernel is still accepted. Unit tests in `dataset/tests/`: canonicalisation on fake jaxprs (fakes pattern from `oracle/tests/test_tpu_oracle.py:13`), assembler on synthetic chains, feedback-verbatim invariant, split-by-family, renderer round trip.

## Why human-written kernels are included

The original draft used scraped human kernels only as reference probes. That was over-cautious. Once every kernel is oracle-verified and contamination-filtered against all 65 exam tasks, a passing human kernel is the highest-quality target available: it is how an expert actually writes Pallas TPU code, and it adds style diversity the constructor alone cannot provide. The real reasons the pool is small are practical, not principled: most public Pallas TPU kernels are the exam's own human solutions (excluded), the remainder need a plain-JAX reference written for them, and they must be rewrapped to satisfy the `workload` / outside-ops contract. Expect 20-40 problems from this source, each yielding up to two examples (human + constructor).

## Handoff notes for the colleague (go in `dataset/README.md`)

- Targets are code only; `meta.plan` is optional context, not in the default render.
- `meta.target_origin` is `constructor` or `human`. Filter on it if you want one style only.
- Repair examples are 3 or 5 turns; the feedback turns are verbatim oracle output and should be rendered exactly as `render_chat.py` does if the eval harness uses the same text.
- `final/reverify_items.jsonl` re-judges the whole set under a newer oracle in one queue job.
- `final/unsolved.jsonl` lists problems no kernel passed; useful as hard-case analysis, not training data.
