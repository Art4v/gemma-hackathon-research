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

Base Gemma solves 0 of 64. Gemini Flash with docs solves TBD(4+) of 64. The LoRA solves TBD.

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

TBD(Hruday):
- LoRA rank, alpha, target modules, learning rate, epochs, maximum sequence length
- Gemma 4 E4B base and the serving stack
- examples rendered with exactly the evaluation's prompt template, with loss on assistant turns only and no example truncated

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
task, against up to 144 in the JAXBench paper. TBD(A2): 3 independent chains per task.

**Analysis.** The primary comparisons are `lora` vs `bare_stack` and `lora` vs `flash_docs`, each a paired test on per-task outcomes.
Nothing is tuned on JAXBench.

## 4. Results

### 4.1 The dataset

TBD (final export). Pre-freeze dry run on Nov 9:
- 172 examples (160 train, 12 val) from 159 solved problems
- by source: 123 authored, 46 KernelBench, 3 scraped
- pass@1 0.87, pass@3 0.95
- 91% of targets at parity (0.97x XLA or faster)
- the median correct kernel runs at 1.5x XLA

### 4.2 Baselines on JAXBench

| Arm | pass@1 | pass within 3 | correct within 3 | Median speed (correct) |
|---|---|---|---|---|
| `bare` (Gemma 4 E4B) | 0/64 | 0/64 | 0 | n/a |
| `docs` (Gemma 4 E4B + 25k docs) | 0/64 | 0/64 | 0 | n/a |
| `flash_docs` (Gemini Flash + docs) | 1/64 | TBD(>= 4)/64 | TBD | TBD (0.31x after turn 0, 0.53x after turn 1) |

**Base Gemma** never gets past the kernel check. Its replies loop to the token cap, misname APIs, or skip `pallas_call`.

**Gemini Flash with docs:**
- 24 of its 64 first attempts are correct, close to JAXBench's 37% per-sample rate with docs, but only one is fast enough to pass.
- It writes the right math with poor tiling: the median correct kernel runs at 0.31x XLA.
- 12 first attempts did the real computation outside `pallas_call`, and the kernel check rejected them.

### 4.3 The fine-tuned model

TBD:
- the main table with all arms on the primary set
- the funnel (passes step 1, reaches the chip, correct, pass)
- success vs attempts
- cost per solved task (local inference vs API)

## 5. Discussion and limitations

- **Timing.** The oracle times with the host clock around `block_until_ready`. JAXBench uses device-side profiler time and notes
  10-20% host overhead on sub-millisecond kernels. Both sides of our ratio pay that overhead, so for tiny kernels `speed_vs_xla` is
  pulled toward 1.0. This makes the 0.7 bar slightly lenient and can inflate the parity share. We report parity split by XLA time.
- **Aliasing.** The oracle doesn't guard `input_output_aliases`. Because it jits `workload` without donation, an aliased
  `pallas_call` writes to an XLA-internal copy, so we don't expect silent corruption, but we haven't proven it.
- **Exclusions.** Items that crash the TPU compiler twice are excluded from the data (TBD(n)) and count as failures in evaluation.
- **Budget.** Our comparisons are at 3 attempts. With far more attempts (JAXBench's 144), stronger models solve almost every task.
  How the gap changes with budget is future work.
- **Repair data.** Most constructor kernels pass at once, so repair examples are few (TBD(n)).

## 6. Conclusion

TBD.

## Reproducibility

Repository: `Art4v/gemma-hackathon-research`.
- **Oracle:** `oracle/` (sha `c9a07d9298a1`)
- **Dataset build:** `dataset/` (branch `dataset-gen`, tag `dataset-v1`)
- **Evaluation:** `control/` (`bench.py`, `bench_gemini.py`, `PROTOCOL.md`)
- **Oracle results:** every result row is on an `oracle-results/*` branch.
