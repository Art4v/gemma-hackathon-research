# Evaluation protocol (pre-registered)

Draft Nov 8, 23:15 UTC, for Hruday to approve. **Once merged, this file is fixed before any LoRA result exists**, and its commit time
is the evidence. A change after that goes in the amendments section with its date and reason, and only if no LoRA result has been
seen yet. Fields marked `TODO(Hruday)` must be filled in before the first LoRA attempt is generated.

## Question

Does a LoRA trained on `dataset-v1` (Claude-constructed, TPU-verified Pallas kernels for problems filtered against all 65 JAXBench
tasks) make Gemma 4 E4B write correct, fast Pallas kernels for JAXBench tasks it never saw? The comparisons are against the same model
without the adapter (`bare`) and with about 25k tokens of Pallas docs in context (`docs`).

## Arms

| Arm | Model | Prompt | Status |
|---|---|---|---|
| `bare` | `gemma4:e4b-it-qat` (Ollama) | the task's plain-JAX reference (`bench.py` `SYSTEM`, `BEFORE_REF`, `AFTER_REF`) | done: 0/64 |
| `docs` | the same | `pallas_guides/teacher_docs.md`, then the same prompt | done: 0/64 |
| `lora` | the same base + the `dataset-v1` LoRA | identical to `bare` | to run |
| `bare_stack` | the LoRA's inference stack with the adapter off | identical to `bare` | to run, **required if the LoRA is served by anything other than the same Ollama model** |

The primary comparison is `lora` vs `bare_stack`, which differ only in the adapter. `lora` vs `bare` and `lora` vs `docs` are also
reported. Inference stack for `lora` (filled Nov 9, before any LoRA attempt):
- **Engine:** Ollama 0.40.14 on Modal (workspace `codecraftacademy981`, A10G, the `control/modal_gemma.py` image), the same engine,
  options and `bench.py` loop as `bare`/`docs`.
- **Weights:** the official `gemma4:e4b-it-qat` GGUF (digest `ee665637121887cf3befff38abbb1be4ee117c7db867d97a67e29049ecd7e15f`, the
  one the `bare`/`docs` rows record), with the LoRA folded in by `training/to_ollama.py`. For each LoRA target tensor: Q4_0(W +
  (alpha/r)·B·A); every other tensor byte-identical.
- **Quantization:** Q4_0, as the official file.
- **Chat template:** Ollama's built-in `gemma4:e4b-it-qat` template, carried over unchanged.
- **`bare_stack`:** the same file with a zero adapter. The pilot's check showed Q4_0(bf16 base) reproduces the official file on 100%
  of blocks, so `bare_stack` is the official model and its rows are the `bare` rows. This is re-verified on the folded `dataset-v1`
  file before eval.

## Tasks

- **Primary set:**
  - the 50 level 1 tasks (`benchmarks/jaxbench/level1/*/baseline.py`, unchanged)
  - the level 2 tasks whose `control/refs` reference matched Google's kernel on the TPU (`verify_refs.py`) before the first LoRA
    attempt is generated. As of Nov 8 that's 51p, 53p, 56p, 57p, 58p, 61p, 62p, 63p, so **58 tasks**.
- **Joining later:** 52p, 60p and 64p join only if they verify before the first LoRA attempt. The amendment lists them with the time.
- **Out:** 54p and 59p (Google's kernel doesn't compile on JAX 0.10.2 / v6e, so the reference can't be checked), 65p (reference out
  of HBM), and 55p (unordered top-k output vs element-wise comparison).
- **Reported every time:** `bare` and `docs` on the same task set as `lora`, recomputed from their existing rows.

## Generation settings (all arms, from `bench.py`, unchanged)

- **Sampling:** Ollama `/api/chat`, thinking off, temperature 0, seed `crc32("<task>/0/<turn>")`, `num_ctx` 65536, at most 8,192
  output tokens, one sample per task.
- **Turns:** turn 0 plus up to 2 retries. A retry sees every earlier attempt with the oracle's `feedback` line, then "Reply with the
  corrected Python code block."
- **Stop rule:** `pass`, `too_large_for_chip` or `harness_oom`, or after turn 2.
- **Code:** the extracted code goes to the oracle as is. A reply with no code block gets the oracle's own verdict.
- A non-Ollama stack must reproduce these settings: same chat template, same stop and length limits. The reply text and the rendered
  prompt are stored per row, so the match can be checked.

## Judging

- **Oracle:** `oracle_sha` `c9a07d9298a1` (`tpu_oracle.py`, `tpu_check.py`, `outside_ops.py` frozen until submission), 3 seeds,
  speed bar 0.7, TPU v6e, JAX 0.10.2. Every row records its `oracle_sha`; a row with any other value is invalid.
- **Step 1 on CPU:** the kernel check (import, trace, `pallas_call` present, outside-op rule) may run on CPU, since it's static and
  gives the same verdict as the TPU. Every attempt that passes step 1 is judged on a v6e.
- **Compiler crash** (the oracle process dies inside XLA or Mosaic with no row): the item is rerun alone once. If it crashes again it's
  recorded as `crash` and counts as not correct, the same as `runtime`, in every arm.

## Metrics

- **Primary:** `pass_within_3`, the tasks with a `pass` by turn 2. `lora` vs `bare_stack` is compared with an exact two-sided McNemar
  test on the paired task outcomes, alpha 0.05. With a 0-pass baseline, at least 6 one-way discordant tasks are needed for
  p < 0.05.
- **Secondary**, per arm (the `bench.py summary` fields):
  - `pass_at_1`, `correct_within_3`, `parity_within_3` (speed_vs_xla >= 0.97), the median `speed_vs_xla` of correct attempts
  - the **funnel** per turn: passes step 1, reaches the chip, compiles and runs, correct, pass
  - verdict transitions between turns (does the arm use the feedback?)
  - the share of replies with no code block
  - prompt tokens and wall time per attempt (the `docs` arm's cost)
- **Intervals:** 95% Wilson intervals for proportions, and a paired bootstrap (10,000 resamples over tasks) for differences between
  arms.

## Training data and model selection

- **Data (corrected Nov 9; the earlier text said Gemini, which the team reversed on Nov 9 01:03 UTC, #3):** the LoRA is trained only
  on `dataset-v1`, tag on `dataset-gen` (commit a0e6770), frozen Nov 9 ~07:05 UTC, `verify_dataset.py` OK, contamination and
  near-dup filters clean.
  - 264 examples (238 train / 26 val; 28 repair chains). Kernels are by **Claude**: `constructor.backend` is `claude_cli` or
    `claude_code_subagent`, per row.
  - It includes the pilot examples first exported as `dataset-v0`.
  - No JAXBench file, reference or kernel is in it.
- **Hyperparameters of the first adapter, `v1_r64`** (a record of that run, not a pre-registration; `training/train_lora.py`, config `r64`):
  - base: `google/gemma-4-E4B-it-qat-q4_0-unquantized` @ `476025a01dbf`, bf16
  - LoRA: rank 64, alpha 128, dropout 0, on attention (q, k, v, o) and MLP (gate, up, down)
  - training: learning rate 1e-4, 3 epochs, gradient accumulation 4, max sequence length 24,576, shuffle seed 0
  - loss only on the final assistant turn (the graded kernel); earlier repair turns are prompt
  - the checkpoint is the end of epoch 3, with no early stopping. Val loss on the 26 val rows is logged, not used to select.
  - checked on `dataset-v1` before training:
    - rows render byte-identical to `bench.conversation` (264/264, feedback verbatim)
    - tokens: max 14,840 (prompt + target), max 5,921 (target)
    - no row over 24,576, and every target within the eval's 8,192 The training examples are rendered with `bench.py`'s exact prompts, with no example truncated and the loss on assistant
  turns only.
- **No adapter is pre-registered as the headline (A5).** The team iterates on the training procedure (data, objective,
  hyperparameters, quantization handling) and may evaluate successive adapters on JAXBench. The eval itself stays fixed: same tasks,
  prompts, decoding, turns and oracle for every arm.
- **Every adapter evaluated on JAXBench gets reported**, in the order it was tried, with its training data, procedure and result, and
  the writeup states how many JAXBench evaluations were run in total. The dataset's val split is used for cheap checks before a
  JAXBench run.

## Invalidating events

Any of these makes a result invalid or reportable only with a caveat:
- an exam file in the training data
- an oracle file change
- a prompt or setting difference between `lora` and `bare_stack` beyond the adapter
- a task-set change after the first LoRA attempt
- an adapter evaluated on JAXBench but left out of the report

## Amendments

**A1 (Nov 9, 01:15 UTC, before any result for this arm): a Gemini Flash reference arm, `flash_docs`.**
- Model `gemini-3.5-flash` on the paid Gemini API (served as `gemini-3.6-flash`), through `control/bench_gemini.py`.
- **Same as the Gemma arms:** the 64 tasks, the docs-arm prompt (`bench.conversation`, word for word), turn 0 plus 2 retries with the
  oracle's feedback, the stop rule and the oracle (`c9a07d9298a1`, 3 seeds, speed bar 0.7).
- **Differs only where needed** to run Gemini at its strongest single-sample setting: thinking on, default temperature, no seed, and a
  64k output budget (thinking counts against it).
- **Reported:** the same metrics as every arm, on the same primary task set. The comparison of interest is `lora` vs `flash_docs` at
  equal attempt budget, one sample and at most 3 attempts. It is stated next to the JAXBench paper's Gemini 3 Flash result, which used
  up to 144 attempts per task (18 chains x 8 turns), so the budgets are not confused.

**A2 (Nov 9, 02:10 UTC; proposed, pending Hruday's OK; written before any `lora` or `flash_docs` verdict): attempt budget and decoding.**
- **Budget, every arm** (`bare`, `docs`, `lora`, `bare_stack`, `flash_docs`): **3 independent chains per task**. A chain is turn 0 plus up to
  2 retries with the oracle's feedback (unchanged), so at most 9 attempts per task. The JAXBench paper's iterative refinement allowed
  up to 144 (18 chains x 8 turns).
- **Decoding:** each model's recommended sampling. Gemma: temperature 1.0, top_p 0.95, top_k 64 (Gemma's documented defaults;
  confirm for Gemma 4). Gemini: API defaults. Seed `crc32("<task>/<chain>/<turn>")` where the engine takes one; chain 0 keeps
  today's seed. Rationale: Gemini 3 is documented to loop at temperature 0, and greedy Gemma loops to the token cap. One decoding
  regime per vendor's guidance treats the arms alike.
- **TPU:** chains are independent, so each turn of all chains is one oracle job, still 3 rounds per arm.
- **Primary metric** (replaces the single-chain `pass_within_3`): an arm's solve rate = mean over tasks of (chains solved within 3
  turns) / 3.
  - **Two pre-specified comparisons:** `lora` vs `bare_stack`, and `lora` vs `flash_docs`. Each uses a two-sided paired permutation
    test on per-task solved counts (10,000 permutations), Holm-corrected, with 95% paired-bootstrap intervals.
- **Secondary:**
  - any of the 9 attempts (pass@3 chains)
  - turn-0 pass@1 (unbiased, from 3 chains)
  - success vs attempt budget (1 to 9)
  - cost per solved task (API dollars vs local)
  - wall time
  - the greedy single-chain `bare`/`docs` results already run, reported as before
- **Fallback if Gemini credit runs short:** 2 chains for every arm. Decided before the first `lora` attempt.
- Flash's turn-0 attempts generated on Nov 9 (API defaults) are its chain 0.

**A3 (Nov 9, 07:10 UTC, before any `lora` verdict): A2 rejected.** Hruday's decision. Every Gemma arm (`bare`, `docs`, `lora`,
`bare_stack`) keeps 1 greedy chain x 3 turns, exactly as the existing bare/docs baselines ran: same Ollama stack and settings. The
primary comparison stays `lora` vs `bare_stack`, an exact McNemar test on per-task solved/not over the task set. `bare_stack` is the
official `gemma4:e4b-it-qat` GGUF with no LoRA folded in (`training/to_ollama.py` folds the adapter into a copy of that GGUF), so in
practice it is the existing `bare` arm. The Gemini Flash arm (A1) samples at its own defaults because Gemini loops at temperature 0;
that asymmetry is stated in the writeup's limitations, next to the equal attempt budget.

**A5 (Nov 9, 16:45 UTC, after the `v1_r64` result): no pre-registered headline adapter.** Hruday did not agree to naming `v1_r64`
as the headline in advance (the TODO was filled in by his agent), so that pre-registration is withdrawn. `v1_r64` (0/64) is reported
as the first iteration. From here the team iterates on the training procedure and may run successive adapters on JAXBench; every one
that is run is reported, with the total number of JAXBench evaluations. Eval settings, tasks and oracle are unchanged.
