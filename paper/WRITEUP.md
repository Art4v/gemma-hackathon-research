# Teaching a 4B Gemma to write TPU kernels

**Subtitle:** a TPU-verified Pallas dataset, an oracle that refuses disguised kernels, and an equal-budget comparison with Gemini Flash

<!-- Draft. Placeholders are TBD(...). Limit: 3,000 words. -->

## Abstract

Custom TPU kernels written in Pallas can beat XLA by large factors, but writing them is specialist work, and even frontier models
struggle with the sparsely documented DSL: JAXBench reports 5.8% per-sample correctness for Gemini 3 Flash without curated
documentation. We ask whether Gemma 4 E4B, a model small enough to run on a laptop, can learn this skill from verified examples.

We build three things:
1. **A TPU oracle** that accepts a solution only if it is a real Pallas kernel, compiles and runs at full size on a TPU v6e, matches
   a float32 answer key on three input seeds, and runs at least 0.7x as fast as XLA.
2. **A dataset** of TBD(N) problems, screened against every JAXBench task by jaxpr op-chain signatures and near-duplicate text, with
   TBD(N) TPU-verified examples, including multi-turn repairs driven by the oracle's verbatim feedback.
3. **An equal-budget evaluation** that gives base Gemma, Gemma with 25k tokens of Pallas docs, Gemma with a LoRA, and Gemini Flash
   with docs the same attempt budget on JAXBench's 64 tasks.

Base Gemma solves 0 of 64 and Gemini Flash with docs solves 8 of 64. The first LoRA also solves 0 of 64. It learns the
format (no repetition loops, the right length, `pallas_call` in every reply) but not Pallas's API semantics, and every attempt fails
the oracle's first check.

## 1. Introduction

Pallas is JAX's kernel language for TPUs. A good Pallas kernel tiles its arrays to fit VMEM, keeps the MXU busy, and fuses operations
XLA would otherwise run separately; Google's paged and ragged attention kernels run 2-5x faster than XLA on a v6e. But Pallas is
young and thinly documented, and its TPU lowering rules (block shapes, layouts, supported primitives) are strict. Models trained
mostly on CUDA and PyTorch hit these rules constantly. In JAXBench, more than half of failed samples, even for the best agent, are
API misuse or compile errors, and MultiKernelBench reports 8-10% pass@1 for the best models on its Pallas tasks.

Fine-tuning small models for GPU kernels (Triton, CUDA) is an active line of work, but to our knowledge nobody has published a
fine-tuned Pallas model, or any result for a small open model on JAXBench. The JAXBench paper names "fine-tuning or reinforcement
learning from execution feedback on curated Pallas corpora" as future work. We take that step with a model that fits on a single
consumer machine.

Our contributions are the oracle, whose kernel check rejected 12 of Gemini Flash's 64 first attempts on JAXBench; a
contamination-screened training set in which every example is a kernel the oracle passed; an equal-budget
evaluation; and TBD(results).

## 2. Related work

**Kernel benchmarks.** KernelBench measures LLM-written CUDA kernels against PyTorch; KernelBench-Verified shows that weak tests let
models reward-hack, cutting the best headline speedup from 1.43x to 0.88x. JAXBench (Tschand, Hong et al., 2026) runs 50 workloads
on a v6e, 17 from MaxText and 33 KernelBench Level 2 fusions; with curated docs, iterative refinement solves 48 of 50 using up to
144 attempts per task. MultiKernelBench covers CUDA, AscendC and Pallas, and MaxKernel is a multi-agent system for TPU kernels.
All of these evaluate prompting or search; none trains a model.

**Training kernel models.** KernelLLM fine-tunes Llama 3.1 8B on PyTorch-Triton pairs; AutoTriton, TritonRL, Dr. Kernel and
DRTriton add RL from execution on 7-8B models; Kevin-32B uses multi-turn RL for CUDA. Several report reward hacking (calling the
reference, skipping the kernel) and add checks against it; our kernel check is the TPU analogue. We use supervised fine-tuning on
oracle-verified trajectories, not RL, to fit a hackathon's compute.

## 3. Method

### 3.1 Task format

A task is a plain-JAX module, exactly as in JAXBench: `CONFIG` holds all sizes, `create_inputs()` returns seeded arrays, and
`workload(*inputs)` is the reference computation. A solution is a module that defines `workload` with the same arguments and does its
main computation inside `pl.pallas_call`.

### 3.2 The oracle

Each solution goes through four steps. The first failure is the verdict.

1. **Is it a kernel?** The solution is traced without running. There must be a `pallas_call`, the output must depend on it, and
   outside it only layout and dtype primitives or small integer metadata are allowed. Otherwise the verdict is `not_a_kernel`.
2. **Does it compile and run?** On a TPU v6e, at the task's full input size. Otherwise `compile`, `vmem`, `mosaic`, `timeout` or
   `runtime`, with the chip's error text as feedback.
3. **Is it correct?** On 3 seeds, against a float32 answer key computed at matmul precision "highest". A seed passes if the output is
   within atol = rtol = 1e-2, or if its error is at most twice XLA's own default-precision error. Seeds 1 and 2 fold the seed into
   every RNG key that `create_inputs` uses. Wrong on any seed gives `wrong`.
4. **Is it fast enough?** `speed_vs_xla = t_xla / t_kernel`, each the median of 20 timed calls after 3 warm-ups, on seed 0. A correct
   solution at 0.7x or faster is `pass`; otherwise it's `correct_slow`.

The speed bar of 0.7 is a policy choice, not a statistical one. Timing noise across sessions is about 3%, so we also report the
**parity** share, solutions at 0.97x or faster. Every row records the hash of the three oracle files (`c9a07d9298a1`, unchanged
across every result in this paper), the seeds, the bar, the chip and the JAX version (0.10.2).

Two CPU-side tools speed things up but never decide a pass. Step 1 is static, so it runs on CPU with an identical verdict. A
cross-lowering filter compiles a kernel for TPU on a CPU host and only ever rejects: on 36 kernels the chip had already judged, it
rejected 28 of the 29 that failed on the chip and none of the 7 that ran.

A handful of kernels crash XLA's TPU compiler itself, killing the oracle process with no verdict. We rerun such an item alone and
record a second crash as a harness failure.

### 3.3 Problems and contamination control

Problems come from three sources:
- KernelBench Level 1 and 2 problems, ported to JAX and checked against the PyTorch module
- model operators written for this project: attention variants (sliding window, sinks, ALiBi, softcapping, MQA decode, chunked
  prefill), MoE routing and dispatch, quantized matmuls (int8, int4, fp8), KV-cache updates, SSM scans, ragged ops
- references for public Pallas kernels from MaxText and tpu-inference, checked against those kernels in interpret mode

Before any kernel is written, every problem passes seven CPU checks at full size: it loads, fits HBM, has deterministic and
seed-varying inputs, runs, and gives finite outputs at default and highest precision.

**Contamination screen.** The exam is all 73 JAXBench files (50 level 1 references and 23 level 2 kernels), screened in three layers:
1. A KernelBench-stem match.
2. A **jaxpr op-chain graph signature.** `jax.make_jaxpr(workload)` on abstract inputs, with sub-jaxprs flattened, layout primitives
   dropped, names canonicalized and repeats collapsed. Equal chains are excluded, and an operation-multiset Jaccard of 0.8 or more
   goes to review.
3. **Near-duplicate text**, on two token streams, one with identifiers kept and one with identifiers abstracted, so renaming can't
   hide a copy.

The final screen kept TBD(378 of 384) problems. We dropped the TBD(3) flagged for review.

### 3.4 Kernel generation and trajectories

A constructor model writes each kernel from a fixed prompt: kernel-writing rules plus about 25k tokens of Pallas docs. Every kernel is
judged by the oracle, and a failing kernel gets at most two repairs, each prompted with its earlier kernels and the oracle's verbatim
feedback line. From these chains, **only passing kernels become training targets.** A chain solved on attempt k becomes one
single-turn example (problem, then the passing kernel) and, for k > 1, one repair example (problem, failed kernel, feedback, and so on
up to the passing kernel); its `correct_slow` kernels appear only as the failing side of a repair. The `v3` base set that `v4` extends
is mixed: of the TBD(2,949) training rows, TBD(2,781) have a target that passed the oracle, TBD(47) carried over from `v3` are correct
but slower than 0.7x XLA, and TBD(121) shape variants and generated kernels were checked only by lowering on CPU.

We chose the constructor by measuring candidates through this same pipeline and oracle:

| Constructor | Round-1 pass (problems) |
|---|---|
| Claude Opus (pipeline) | 91% (108 of 119), and 117 of 119 within 3 attempts |
| Gemini 3.5 Flash-Lite | 16% (6 of 38) and 9% (4 of 47) |
| Gemini Flash (served as 3.6) | 35% (7 of 20) on a matched 20-problem sample, against 20% for Flash-Lite |

TBD: the constructor note for the final dataset.

### 3.5 Training

We fine-tune Gemma 4 E4B with LoRA on `dataset-v1` (adapter `v1_r64`): rank 64 and alpha 128 on the attention and MLP projections
(139.5M trainable parameters), learning rate 1e-4, gradient accumulation 4, 3 epochs (714 steps), seed 0, no early stopping, and loss
on the final assistant turn only. Training took about 20 minutes on one GPU. Validation loss reached its minimum of 0.350 at step 400
and ended at 0.354, flat with no sign of overfitting.

Training rows are built by the evaluation's own prompt function, so they are **byte-identical** to what the model sees at test time
(an independent rebuild gives an empty diff). No row is truncated (maximum 6,957 tokens against a 24,576 cap), and every target fits
the evaluation's 8,192-token output limit.

**Serving.** Ollama does not load LoRA adapters, so the adapter is folded into a copy of the same 4-bit QAT GGUF
(`gemma4:e4b-it-qat`, digest `ee665637...`) that the baselines used. Ollama's chat template and the training (HF) template give
identical prompt token counts on 6 rows covering single-turn and one- and two-repair chains, so the comparison has a single
difference: the adapter.

### 3.6 Evaluation protocol

The protocol was committed before any LoRA result (`control/PROTOCOL.md`).

**Tasks.** JAXBench's 50 level 1 tasks plus 14 level 2-only tasks, for which we wrote plain-JAX references and checked them on the
TPU against Google's kernels. The primary set is the 58 tasks whose references verified.

**Arms.**

| Arm | Model | Prompt |
|---|---|---|
| `bare` | Gemma 4 E4B | the reference only |
| `docs` | Gemma 4 E4B | the same Pallas docs the constructor saw, then the reference |
| `lora` | Gemma 4 E4B + adapter | the reference only |
| `bare_stack` | the adapter's serving stack, adapter off | the reference only |
| `flash_docs` | Gemini Flash, with docs, at its default thinking and temperature | the docs-arm prompt |

**Budget.** Every arm gets a first attempt plus two repairs, each seeing the oracle's feedback: 3 attempts per task, against up to
144 in the JAXBench paper. Gemma arms decode greedily, one chain per task.

**Analysis.** The primary comparisons are `lora` vs `bare_stack` and `lora` vs `flash_docs`, each a paired test on per-task outcomes.
Nothing is tuned on JAXBench.

## 4. Results

### 4.1 The dataset

The `dataset-v1` export (Nov 9) has **264 examples** (238 train, 26 val, split by family) from **236 solved problems** out of 255
judged:
- 236 single-turn examples and **28 repair chains** (22 with one repair, 6 with two)
- by source: 123 authored, 138 KernelBench, 3 scraped
- constructor pass@1 0.82, pass@2 0.91, pass@3 0.93 (within 3: authored 0.98, KernelBench 0.90)
- 83% of targets at parity, and round-1 correct kernels at a median of 1.29x XLA (Figure 2)
- KernelBench convolutions are the hard case: 67% pass and 31% are correct but slower than 0.7x, since XLA's convolutions are
  strong on the v6e (Figure 3)

### 4.2 Baselines on JAXBench

| Arm | pass@1 | pass within 3 | correct within 3 | Median speed (correct) |
|---|---|---|---|---|
| `bare` (Gemma 4 E4B) | 0/64 | 0/64 | 0 | n/a |
| `docs` (Gemma 4 E4B + 25k docs) | 0/64 | 0/64 | 0 | n/a |
| `flash_docs` (Gemini Flash + docs) | 1/64 | 8/64 | 37/64 | 0.41x |
| `lora` (Gemma 4 E4B + `v1_r64`) | 0/64 | 0/64 | 0 | n/a |

**Base Gemma** never gets past the kernel check: its replies loop to the token cap, misname APIs, or skip `pallas_call`.

**Gemini Flash with docs** gets 24 of its 64 first attempts correct, close to JAXBench's 37% per-sample rate with docs, but only one
is fast enough to pass. It writes the right math with poor tiling: the median correct kernel runs at 0.31x XLA. The kernel check
rejected 12 first attempts that did the real computation outside `pallas_call`.

### 4.3 The fine-tuned model

**The first adapter, `v1_r64`, solves 0 of 64.** That holds on the primary set (58 tasks) and on the S-p subset (25
verified p-tasks), and against `bare_stack` (the same GGUF without the adapter) there are no discordant tasks. All 192 attempts were
decided at the oracle's first check (trace and kernel check, on CPU):

| Verdict | turn 0 | turn 1 | turn 2 |
|---|---|---|---|
| `compile` | 61 | 63 | 64 |
| `not_a_kernel` | 3 | 1 | 0 |
| passes step 1 | 0 | 0 | 0 |

**The adapter learned the format.** No reply loops to the token cap (the base model's main failure), reply lengths match the
training targets (1.2-1.8k tokens), and every reply contains a fenced module with `pallas_call`. **It did not learn the API's
semantics.** Its errors include index maps that return slices (`lambda b, h: (b, h, :, :)`, a syntax error), grid and index-map
arity mismatches, and `out_specs` / `out_shape` structure mismatches. The oracle's verbatim feedback on these does not lead to a fix
in the next turn.

**Most of the adapter never reached the served model.** After the JAXBench run we checked the fold. The LoRA was trained against
bf16 weights and then rounded into Q4_0. The median LoRA change to a weight is 2% of one Q4_0 rounding step, and none exceeds half
a step, so rounding keeps only a sparse, random part of the adapter. An earlier check (93.5% of blocks changed) measured changed
bytes, not surviving effect, and missed this. A mixed-precision fold fixes it: the 258 LoRA-touched tensors are stored at F16 as
`dequant(Q4_0) + delta`, and every other tensor stays byte-identical. On the 23 val problems:

| `v1_r64` served as | uses `ShapeDtypeStruct` | median similarity to target | passes step 1 (CPU) |
|---|---|---|---|
| Q4_0 fold (the JAXBench run above) | 6/23 | 0.28 | 1/23 |
| bf16 adapter in HF (reference) | 19/23 | 0.47 | 4/23 |
| F16 fold in Ollama | 18/23 | 0.51 | 3/23 |

The F16 fold matches the unquantized adapter, so the serving bug is fixed. But even intact, `v1_r64` gets only 3 of 23 val
kernels past step 1. A second adapter trained longer on the same data (`v2`) reached a lower training loss (0.07 vs 0.18) and a
higher val loss (0.35 to 0.47): with 238 training rows it memorizes. Published kernel models trained on 2k to 60k verified examples.

**Deviation from the protocol.** The serving bug was found after the first JAXBench result, so `v1_r64` at 0/64 stays reported as
the first iteration. Later adapters change the data and the serving fold, are selected on val only, and are all reported with
the total number of JAXBench runs (amendment A5).

TBD: the `v3` adapter (dataset-v1 plus about 1.2k injected-error repair pairs and about 1.5k CPU-verified generated kernels, trained
with MLX on the exact Q4_0 weights), its val numbers, and its JAXBench result.

## 5. Discussion and limitations

- **Timing.** We time with the host clock around `block_until_ready`; JAXBench uses device time and notes 10-20% host overhead on
  sub-millisecond kernels. Both sides of our ratio pay it, pulling tiny kernels toward 1.0x, so the 0.7 bar is slightly lenient
  and we report parity split by XLA time.
- **Aliasing.** The oracle doesn't guard `input_output_aliases`. It jits `workload` without donation, so an aliased `pallas_call`
  writes to an XLA-internal copy; we expect no silent corruption but haven't proven it.
- **Exclusions.** Items that crash the TPU compiler twice are excluded from the data (TBD(n)) and count as failures in evaluation.
- **Budget and decoding.** Our comparisons are at 3 attempts; with far more (JAXBench's 144), stronger models solve almost every
  task, and how the gap changes with budget is future work. The budget is equal but the decoding is not: Gemma arms decode greedily
  (temperature 0, one chain), where small models can repeat until the token cap, while Gemini Flash samples at its own defaults
  because it loops at temperature 0.
- **Repair turns.** v1's repair turns never fixed a kernel, matching a study of 1.5B to 7B code models where the model's own failed
  attempt cost 6.1 points at 1.5B and execution feedback did no better than a placebo (arXiv 2607.26117). We keep repairs so every
  arm sees the same budget, which may understate small models.
- **Quantized serving.** Folding a bf16-trained LoRA into a Q4_0 GGUF erased most of it (Section 4.3). Anyone shipping a LoRA on a
  4-bit on-device model should check the effect that survives the fold, or train on the quantized weights directly.
- **Repair data.** Most constructor kernels pass at once, so repair examples are few (TBD(n)).

## 6. Conclusion

TBD.

## Reproducibility

Repository: `Art4v/gemma-hackathon-research`.
- **Oracle:** `oracle/` (sha `c9a07d9298a1`)
- **Dataset build:** `dataset/` (branch `dataset-gen`, tag `dataset-v1`)
- **Evaluation:** `control/` (`bench.py`, `bench_gemini.py`, `PROTOCOL.md`)
- **Oracle results:** every result row is on an `oracle-results/*` branch.
