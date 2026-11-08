# Evaluation protocol (pre-registered)

Draft Nov 8, 23:15 UTC, for Hruday to approve. **Once merged, this file is fixed before any LoRA result exists**, and its commit time
is the evidence. A change after that goes in the amendments section with its date and reason, and only if no LoRA result has been
seen yet. Fields marked `TODO(Hruday)` must be filled in before the first LoRA attempt is generated.

## Question

Does a LoRA trained on `dataset-v1` (Gemini-constructed, TPU-verified Pallas kernels for problems filtered against all 65 JAXBench
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
reported. Inference stack for `lora`: `TODO(Hruday)` (engine, weights file, quantization, chat template source).

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

- **Data:** the LoRA is trained only on `dataset-v1`: Gemini constructor, frozen Nov 11 12:00 UTC, `verify_dataset.py` passing. No
  Claude-generated data (`dataset-v0`) and no JAXBench file, reference or kernel.
- **Hyperparameters, fixed before training:** `TODO(Hruday)` (rank, alpha, target modules, learning rate, epochs, max sequence length,
  seed). The training examples are rendered with `bench.py`'s exact prompts, with no example truncated and the loss on assistant
  turns only.
- **No selection on JAXBench:** no hyperparameter, checkpoint, epoch, prompt or decoding choice is made using any JAXBench result. If
  something has to be chosen, it's chosen on the dataset's own val split.
- **Every LoRA trained gets reported**, each with its training-set size (e.g. a data-size curve). The headline LoRA is named here in
  advance: `TODO(Hruday)` (expected: the full `dataset-v1` adapter).

## Invalidating events

Any of these makes a result invalid or reportable only with a caveat:
- an exam file in the training data
- an oracle file change
- a prompt or setting difference between `lora` and `bare_stack` beyond the adapter
- a task-set change after the first LoRA attempt
- a LoRA chosen using JAXBench outcomes

## Amendments

(none yet)
