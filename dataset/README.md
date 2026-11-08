# dataset

A LoRA training set that teaches Gemma 3n E4B to write Pallas TPU kernels, built and verified against the oracle in `oracle/`.
`PLAN.md` holds the decisions and the two phases; `PROCESS.md` is the step-by-step build; this file is the handoff.

## What a record is

`final/train.jsonl` and `final/val.jsonl` hold training examples. Each has `turns`:

| turn | role | type | payload |
|---|---|---|---|
| 1 | user | `problem` | `ref_code`: a plain-JAX baseline module (`CONFIG`, `create_inputs()`, `workload`) |
| 2 | assistant | `kernel` | `code`: a Python module defining `workload` with `pl.pallas_call`; `candidate_id` |
| 3, 5 | user | `oracle_feedback` | `feedback`: the oracle's one-line verdict, verbatim; `verdict`; `verdict_ref {job_id, oracle_sha}` |
| 4, 6 | assistant | `kernel` | the repaired module |

`kind` is `single` (2 turns) or `repair` (4 or 6 turns). Every final kernel has an oracle row with verdict `pass` (correct on 3 seeds,
`speed_vs_xla >= 0.7`). `at_parity` marks kernels at `>= 0.97x` XLA. `meta.target_origin` is `constructor` (model-written) or `human`
(a public kernel, oracle-verified). `meta.plan` is the constructor's short plan; it is metadata, not part of the target.

Render with `scripts/render_chat.py`: user content = fixed instruction + fenced reference; assistant = fenced code only; feedback turn
= `Oracle verdict: <feedback>` + fixed retry instruction. If the eval harness uses the same wording, train and eval match.

## Other files in `final/`

- `stats.json`: counts by source, kind, repairs, origin; pass@1/2/3 per source; speed quantiles; parity share.
- `MANIFEST.json`: sha256 of every file, the oracle commit/sha set, prompt versions.
- `reverify_items.jsonl`: every final kernel as an oracle item. Submit it as one queue job to re-judge the set under a newer oracle.
- `unsolved.jsonl`: problems no kernel passed in three attempts. Hard-case analysis, not training data.

## Splits

~10 percent of problem families go to `val`, by hash of `family_id`; shape variants of one base problem never straddle the split.

## Invariants (checked by `scripts/verify_dataset.py`)

- Every kernel in `final/` has a matching oracle row that says `pass` (ids, task, reference sha and code sha all match).
- Every feedback turn is the oracle's `feedback` for the kernel in the preceding turn, byte for byte.
- No JAXBench reference or kernel, no KernelBench problem the exam uses, and no exam op-chain appears in `final/`.
- Every row was judged with `speed_bar` 0.7 and seeds `[0, 1, 2]`.

## Running the pipeline

```bash
bash scripts/env.sh && source .venv/bin/activate         # Python 3.12 venv with JAX 0.10.2 CPU and torch
python scripts/fetch_sources.py                           # KernelBench, jax, MaxText, tpu-inference into raw/
python scripts/port_kernelbench.py --pick easy --limit 20 # constructor ports PyTorch problems; torch equivalence check
python scripts/author_problems.py --limit 10              # constructor authors p-style problems
python scripts/wrap_scraped.py --limit 5                  # public kernels -> reference + human candidate
python scripts/filter_contamination.py && python scripts/filter_neardup.py
python scripts/generate.py --batch pilot --round 1        # constructor writes kernels (claude -p by default)
python scripts/prescreen.py --batch pilot --round 1       # parse + the oracle's --static-only check
python scripts/oracle_submit.py --batch pilot --round 1 --probes [--enqueue]   # items + request; --enqueue copies to oracle/queue
# ... operator runs the job on the v6e, commits oracle/results/<job_id>/ ...
python scripts/oracle_collect.py dataset-pilot-r1        # verdicts/ and trajectories/
python scripts/generate.py --batch pilot --round 2 && ... # repair rounds 2 and 3
python scripts/assemble.py --phase pilot && python scripts/verify_dataset.py final/pilot
```

Tests: `SERA_JAX_PY=$PWD/.venv/bin/python .venv/bin/python -m unittest discover -s tests -t ..` from `dataset/`.
