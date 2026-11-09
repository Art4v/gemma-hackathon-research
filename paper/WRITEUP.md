# Teaching a 4B Gemma to write TPU kernels

**Subtitle:** a TPU-verified Pallas dataset, an oracle that refuses disguised kernels, and an equal-budget comparison with Gemini Flash

> Draft, Nov 9 (Karthik's agent). Placeholders are marked `TBD(...)`. Target: Kaggle Writeup, Gemma 4 Developer Agent Paper Track,
> 3,000 words maximum. About 2,300 words before results.

## Abstract

Custom TPU kernels written in Pallas can beat XLA by large factors, but writing them is specialist work, and even frontier models
struggle with a sparsely documented DSL. The JAXBench authors report 5.8% per-sample correctness for Gemini 3 Flash without curated
documentation. We ask whether a model small enough to run on a laptop, Gemma 4 E4B, can learn this skill from a few hundred verified
examples.

We build three things:
1. **A TPU oracle.** It accepts a solution only if the solution is a real Pallas kernel, compiles and runs at full size on a TPU v6e,
   matches a float32 answer key on three input seeds, and runs at least 0.7x as fast as XLA.
2. **A dataset.** TBD(N) problems are screened against every JAXBench task by jaxpr op-chain graphs and near-duplicate text, and
   TBD(N) TPU-verified examples include multi-turn repairs driven by the oracle's verbatim feedback.
3. **A pre-registered evaluation.** Base Gemma, Gemma with 25k tokens of Pallas docs, Gemma with a LoRA, and Gemini Flash with docs
   all get the same attempt budget on JAXBench's 64 tasks.

Base Gemma solves 0 of 64 and Gemini Flash with docs solves 8 of 64. The pre-registered LoRA also solves 0 of 64. It learns the format: no repetition loops, the right length, `pallas_call` in every reply. It does not learn Pallas's API semantics, and every attempt fails the oracle's first check.

## 1. Introduction

Pallas is JAX's kernel language for TPUs. A good Pallas kernel tiles its arrays to fit VMEM, keeps the MXU busy, and fuses operations
XLA would otherwise run separately. Google's own kernels for paged and ragged attention run 2-5x faster than XLA on a v6e. But Pallas
is young and thinly documented, and its TPU lowering rules (block shapes, layouts, supported primitives) are strict. Models trained
mostly on CUDA and PyTorch hit these rules constantly. The JAXBench paper reports that more than half of failed samples, even for its
best agent, are API misuse or compile errors. MultiKernelBench reports 8-10% pass@1 for the best models on its Pallas tasks.

Fine-tuning small models for GPU kernels (Triton, CUDA) is an active line of work. Nobody, to our knowledge, has published a
fine-tuned Pallas model, or any result for a small open model on JAXBench. The JAXBench paper names "fine-tuning or reinforcement
learning from execution feedback on curated Pallas corpora" as future work. We take that step with a model that fits on a single
consumer machine.

Our contributions:
1. **A four-step oracle on a real TPU v6e.** Its kernel check rejects solutions that do the real work outside `pallas_call`. It
   rejected 12 of Gemini Flash's 64 first attempts on JAXBench.
2. **A contamination-screened, TPU-verified training set** built from KernelBench ports, model operators written for this project,
   and wrapped public kernels. Every example is a kernel the oracle passed, and repair examples carry the oracle's feedback verbatim.
3. **A pre-registered evaluation** that gives every model the same attempt budget, including a reference arm with Gemini Flash.
4. **Results:** TBD.

## 2. Related work

**Kernel benchmarks.**
- KernelBench measures LLM-written CUDA kernels against PyTorch.
- KernelBench-Verified shows that narrow test inputs and weak baselines let models reward-hack. The best model's 1.43x headline
  speedup drops to 0.88x under a stricter protocol.
- JAXBench (Tschand, Hong et al., 2026) is TPU-native: 50 workloads on a v6e, 17 production operators from MaxText and 33 KernelBench
  Level 2 fusions. It evaluates Gemini with best-of-N, iterative refinement and Autocomp's beam search. With curated docs, iterative
  refinement solves 48 of 50 tasks, using up to 144 attempts per task.
- MultiKernelBench covers CUDA, AscendC and Pallas.
- MaxKernel builds a multi-agent system for TPU kernels.

All of these evaluate prompting or agentic search; none trains a model.

**Training kernel models.**
- KernelLLM fine-tunes Llama 3.1 8B on PyTorch-Triton pairs.
- AutoTriton, TritonRL, Dr. Kernel and DRTriton combine supervised fine-tuning with reinforcement learning from execution, on 7-8B
  models.
- Kevin-32B uses multi-turn RL with execution feedback for CUDA.

Several of these papers report reward hacking, such as calling the reference or skipping the kernel, and add checks against it. Our
oracle's kernel check is the TPU analogue. We use supervised fine-tuning on oracle-verified trajectories, not RL, to stay within a
hackathon's compute.

## 3. Method

### 3.1 Task format

A task is a plain-JAX module, exactly as in JAXBench:
- `CONFIG` holds all sizes
- `create_inputs()` returns seeded arrays
- `workload(*inputs)` is the reference computation

A solution is a module that defines `workload` with the same arguments and does its main computation inside `pl.pallas_call`.

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

**The speed bar** of 0.7 is a policy choice, not a statistical one. Timing noise across sessions is about 3%, so we also report the
**parity** share, solutions at 0.97x or faster.

**Provenance.** Every row records the hash of the three oracle files (`c9a07d9298a1`, unchanged across every result in this paper),
the seeds, the bar, the chip and the JAX version (0.10.2).

**Two CPU-side tools speed things up but never decide a pass.**
- Step 1 is static, so it runs on CPU with an identical verdict.
- A cross-lowering filter compiles a kernel for TPU on a CPU host and only ever rejects. On 36 kernels the chip had already judged,
  it rejected 28 of the 29 that failed on the chip and none of the 7 that ran.

**Robustness.** A handful of kernels crash XLA's TPU compiler itself, killing the oracle process with no verdict. We isolate the
item, rerun it alone, and record a second crash as a harness failure.

### 3.3 Problems and contamination control

**Sources:**
- KernelBench Level 1 and 2 problems, ported to JAX and checked against the PyTorch module
- model operators written for this project: attention variants (sliding window, sinks, ALiBi, softcapping, MQA decode, chunked
  prefill), MoE routing and dispatch, quantized matmuls (int8, int4, fp8), KV-cache updates, SSM scans, ragged ops
- references for public Pallas kernels from MaxText and tpu-inference, checked against those kernels in interpret mode

**Readiness.** Every problem passes seven CPU checks at full size before any kernel is written: it loads, fits HBM, has deterministic
and seed-varying inputs, runs, and gives finite outputs at default and highest precision.

**Contamination screen.** The exam is all 73 JAXBench files (50 level 1 references and 23 level 2 kernels). Three layers:
1. A KernelBench-stem match.
2. A **jaxpr op-chain graph signature.** `jax.make_jaxpr(workload)` on abstract inputs, with sub-jaxprs flattened, layout primitives
   dropped, names canonicalized and repeats collapsed. Equal chains are excluded, and an operation-multiset Jaccard of 0.8 or more
   goes to review.
3. **Near-duplicate text.** Two token streams, one with identifiers kept and one with identifiers abstracted, so renaming can't hide a
   copy.

The final screen kept TBD(378 of 384) problems. We dropped the TBD(3) flagged for review.

### 3.4 Kernel generation and trajectories

A constructor model writes each kernel from a fixed prompt: kernel-writing rules plus about 25k tokens of Pallas docs. Every kernel is
judged by the oracle. A failing kernel gets at most two repairs, each prompted with its earlier kernels and the oracle's verbatim
feedback line. **The training targets are passing kernels only.** A chain solved on attempt k becomes:
- one single-turn example: problem, then the passing kernel
- for k > 1, one repair example: problem, failed kernel, feedback, and so on up to the passing kernel

`correct_slow` kernels appear only as the failing side of a repair.

**Choosing the constructor.** We measured candidates through this same pipeline and oracle:

| Constructor | Round-1 pass (problems) |
|---|---|
| Claude Opus (pipeline) | 91% (108 of 119), and 117 of 119 within 3 attempts |
| Gemini 3.5 Flash-Lite | 16% (6 of 38) and 9% (4 of 47) |
| Gemini Flash (served as 3.6) | 35% (7 of 20) on a matched 20-problem sample, against 20% for Flash-Lite |

TBD: the constructor note for the final dataset.

### 3.5 Training

We fine-tune Gemma 4 E4B with LoRA on `dataset-v1` (adapter `v1_r64`):
- rank 64, alpha 128, on the attention and MLP projections: 139.5M trainable parameters
- learning rate 1e-4, gradient accumulation 4, 3 epochs (714 steps), seed 0, no early stopping
- loss on the final assistant turn only

Training took about 20 minutes on one GPU (peak memory 31.5 GB). Validation loss reached its minimum of 0.350 at step 400 and
ended at 0.354, so it was flat with no sign of overfitting.

Training rows are built by the evaluation's own prompt function, so they are **byte-identical** to what the model sees at test time.
We checked all all rows against an independent rebuild, and the diff is empty. The repair feedback is the oracle's line, verbatim.

Lengths, with the Gemma 4 E4B tokenizer:
- prompt plus target: median 2,140 tokens, maximum 6,957, so nothing is truncated at the 24,576-token cap
- target alone: maximum 5,921 tokens, so every target fits the evaluation's 8,192-token output limit

**Serving.** Ollama does not load LoRA adapters, so the adapter is folded into a copy of the same 4-bit QAT GGUF
(`gemma4:e4b-it-qat`, digest `ee665637...`) that the baselines used. We also checked that Ollama's chat template and the training (HF) template give identical prompt token counts on 6 rows covering single-turn and one- and two-repair chains. That keeps the comparison to a single difference, the adapter.

### 3.6 Evaluation protocol (pre-registered)

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

**Budget.** Every arm gets the same budget: a first attempt plus two repairs, each seeing the oracle's feedback. That's 3 attempts per
task, against up to 144 in the JAXBench paper. Gemma arms decode greedily, one chain per task.

**Analysis.** The primary comparisons are `lora` vs `bare_stack` and `lora` vs `flash_docs`, each a paired test on per-task outcomes.
Nothing is tuned on JAXBench.

## 4. Results

### 4.1 The dataset

The final export (`dataset-gen`, Nov 9) has **264 examples** (238 train, 26 val, split by family) from **236 solved problems** out
of 255 judged:
- 236 single-turn examples and **28 repair chains** (22 with one repair, 6 with two)
- by source: 123 authored, 138 KernelBench, 3 scraped
- constructor pass@1 0.82, pass@2 0.91, pass@3 0.93 (authored 0.98, KernelBench 0.90 within 3)
- 83% of targets at parity (0.97x XLA or faster)
- round-1 correct kernels run at a median of 1.29x XLA (Figure 2)
- KernelBench convolutions are the hard case: 67% pass and 31% are correct but slower than 0.7x, since XLA's convolutions are
  strong on the v6e (Figure 3)

### 4.2 Baselines on JAXBench

| Arm | pass@1 | pass within 3 | correct within 3 | Median speed (correct) |
|---|---|---|---|---|
| `bare` (Gemma 4 E4B) | 0/64 | 0/64 | 0 | n/a |
| `docs` (Gemma 4 E4B + 25k docs) | 0/64 | 0/64 | 0 | n/a |
| `flash_docs` (Gemini Flash + docs) | 1/64 | 8/64 | 37/64 | 0.41x |
| `lora` (Gemma 4 E4B + `v1_r64`) | 0/64 | 0/64 | 0 | n/a |

**Base Gemma** never gets past the kernel check. Its replies loop to the token cap, misname APIs, or skip `pallas_call`.

**Gemini Flash with docs:**
- 24 of its 64 first attempts are correct, close to JAXBench's 37% per-sample rate with docs, but only one is fast enough to pass.
- It writes the right math with poor tiling: the median correct kernel runs at 0.31x XLA.
- 12 first attempts did the real computation outside `pallas_call`, and the kernel check rejected them.

### 4.3 The fine-tuned model

**The pre-registered adapter, `v1_r64`, solves 0 of 64.** That holds on the primary set (58 tasks) and on the S-p subset (25
verified p-tasks). Against `bare_stack` (the same GGUF without the adapter) there are no discordant tasks.

| Verdict | turn 0 | turn 1 | turn 2 |
|---|---|---|---|
| `compile` | 61 | 63 | 64 |
| `not_a_kernel` | 3 | 1 | 0 |
| passes step 1 | 0 | 0 | 0 |

All 192 attempts were decided at the oracle's first check (trace and kernel check, on CPU).

**The adapter learned the format.** No reply loops to the token cap (the base model's main failure), reply lengths match the
training targets (1.2-1.8k tokens), and every reply contains a fenced module with `pallas_call`.

**It did not learn the API's semantics.** Its errors include:
- index maps that return slices (`lambda b, h: (b, h, :, :)`, a syntax error)
- grid and index-map arity mismatches
- `out_specs` / `out_shape` structure mismatches

The oracle's verbatim feedback on these does not lead to a fix in the next turn.

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

**Deviation from the protocol.** The serving bug was found after the pre-registered result, so `v1_r64` at 0/64 stays reported as
the first iteration. Later adapters change the data and the serving fold, are selected on val only, and are all reported with
the total number of JAXBench runs (amendment A5).

TBD: the `v3` adapter (dataset-v1 plus about 1.2k injected-error repair pairs and about 1.5k CPU-verified generated kernels, trained
with MLX on the exact Q4_0 weights), its val numbers, and its JAXBench result.

## 5. Discussion and limitations

- **Timing.** The oracle times with the host clock around `block_until_ready`. JAXBench uses device-side profiler time and notes
  10-20% host overhead on sub-millisecond kernels. Both sides of our ratio pay that overhead, so for tiny kernels `speed_vs_xla` is
  pulled toward 1.0. This makes the 0.7 bar slightly lenient and can inflate the parity share. We report parity split by XLA time.
- **Aliasing.** The oracle doesn't guard `input_output_aliases`. Because it jits `workload` without donation, an aliased
  `pallas_call` writes to an XLA-internal copy, so we don't expect silent corruption, but we haven't proven it.
- **Exclusions.** Items that crash the TPU compiler twice are excluded from the data (TBD(n)) and count as failures in evaluation.
- **Budget.** Our comparisons are at 3 attempts. With far more attempts (JAXBench's 144), stronger models solve almost every task.
  How the gap changes with budget is future work.
- **Decoding asymmetry.** Gemma arms decode greedily (temperature 0, one chain), while Gemini Flash samples at its own defaults,
  because Gemini loops at temperature 0. The attempt budget is equal, but the decoding is not.
- **Greedy loops.** Greedy small models can repeat until the token cap; this was the pilot LoRA's main failure mode.
- **Quantized serving.** Folding a bf16-trained LoRA into a Q4_0 GGUF erased most of it (Section 4.3). Anyone shipping a LoRA
  on a 4-bit on-device model should check the effect that survives the fold, not just which bytes changed, or train on the
  quantized weights directly.
- **Repair data.** Most constructor kernels pass at once, so repair examples are few (TBD(n)).

## 6. Conclusion

TBD.

## Reproducibility

Repository: `Art4v/gemma-hackathon-research`.
- **Oracle:** `oracle/` (sha `c9a07d9298a1`)
- **Dataset build:** `dataset/` (branch `dataset-gen`, tag `dataset-v1`)
- **Evaluation:** `control/` (`bench.py`, `bench_gemini.py`, `PROTOCOL.md`)
- **Oracle results:** every result row is on an `oracle-results/*` branch.
