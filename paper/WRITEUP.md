# Teaching a 4B Gemma to write TPU kernels

**Subtitle:** an oracle-verified Pallas dataset from an open-weights teacher, and a 4B model that passes as many JAXBench tasks as
Gemini Flash with docs

<!-- Draft v7 (Nov 12). Every JAXBench number is recomputed by paper/analysis/jaxbench.py from paper/data/. Limit: 3,000 words. -->

## Abstract

Pallas kernels can beat XLA on TPUs by large factors, but writing them is specialist work, and frontier models struggle with the
thinly documented DSL. We ask whether Gemma 4 E4B, a model small enough to run on a laptop, can learn it from verified examples.

We build:
1. **A TPU oracle.** It accepts a solution only if it is a real Pallas kernel, compiles and runs at full size on a TPU v6e, matches a
   float32 answer key on three seeds, and runs at least 0.7x as fast as XLA.
2. **A training set of 2,304 rows.** Every target was judged correct by that oracle, and 85% also meet its speed bar. The kernels
   come from an open-weights teacher (Kimi K3.1)
   through up to three rounds of oracle feedback, plus a template generator for kernels at JAXBench's sizes. Every row is screened
   against JAXBench at the instance level.
3. **An evaluation** on JAXBench's 64 tasks. Every model gets the same three attempts per task.

On the 58-task primary set, the final adapter (v7) passes **9 tasks within three attempts**. Base Gemma passes 0, our earlier
adapters pass 0 or 1, and Gemini Flash with 25k tokens of Pallas docs passes 7. All nine v7 kernels run faster than XLA
(1.30-1.39x). Eight are matmul fusions, where every earlier Gemma model passed none, because their kernels did not fit in VMEM.

## 1. Introduction

Pallas is JAX's kernel language for TPUs. A good kernel tiles its arrays to fit VMEM, keeps the MXU busy, and fuses operations XLA
would run separately. But Pallas is young, and its TPU lowering rules (block shapes, layouts, supported primitives) are strict. In
JAXBench, more than half of failed samples, even for the best agent, are API misuse or compile errors. The JAXBench authors name
"fine-tuning or reinforcement learning from execution feedback on curated Pallas corpora" as future work; to our knowledge no
fine-tuned Pallas model, and no result for a small open model on JAXBench, has been published.

We fine-tune Gemma 4 E4B with LoRA on oracle-verified kernels and report every JAXBench run we made, failures included. Our
contributions:
- an oracle whose kernel check rejected 12 of Gemini Flash's 64 first attempts on JAXBench;
- a 2,304-row training set from an open-weights teacher and a template generator, every target TPU-verified, screened against
  JAXBench at the instance level;
- the finding that a 4-bit served LoRA must be trained on, and folded into, the exact quantized weights it will run on;
- a 4B model that matches Gemini Flash with docs on fast, correct kernels.

## 2. Related work

**Kernel benchmarks.** KernelBench measures LLM-written CUDA kernels against PyTorch; KernelBench-Verified shows that weak tests let
models reward-hack. JAXBench runs 50 workloads on a v6e (17 from MaxText, 33 KernelBench Level 2 fusions); with curated docs,
iterative refinement solves 48 of 50 using up to 144 attempts per task. MultiKernelBench covers CUDA, AscendC and Pallas. All of
these evaluate prompting or search; none trains a model.

**Training kernel models.** KernelLLM fine-tunes Llama 3.1 8B on PyTorch-Triton pairs; AutoTriton, TritonRL and Dr. Kernel add RL
from execution on 7-8B models; Kevin-32B uses multi-turn RL for CUDA. Several report reward hacking and add checks against it; our
kernel check is the TPU analogue. We use supervised fine-tuning on oracle-verified kernels, which fits a hackathon's compute.

## 3. Method

### 3.1 Task format

A task is a plain-JAX module, as in JAXBench. `CONFIG` holds all sizes, `create_inputs()` returns seeded arrays, and
`workload(*inputs)` is the reference. A solution defines `workload` with the same arguments and does its main computation inside
`pl.pallas_call`.

### 3.2 The oracle

Each solution goes through four steps; the first failure is the verdict.
1. **Is it a kernel?** Traced without running: there must be a `pallas_call`, the output must depend on it, and outside it only
   layout and dtype primitives or small integer metadata are allowed. Otherwise `not_a_kernel`.
2. **Does it compile and run** on a TPU v6e at the task's full size? Otherwise `compile`, `vmem`, `mosaic`, `timeout` or `runtime`,
   with the chip's error text as feedback.
3. **Is it correct** on 3 seeds against a float32 answer key at matmul precision "highest" (atol = rtol = 1e-2, or at most twice
   XLA's own error)? Otherwise `wrong`.
4. **Is it fast enough?** `speed_vs_xla = t_xla / t_kernel` (medians of 20 timed calls). Correct at 0.7x or faster is `pass`;
   otherwise `correct_slow`.

Every result in this paper uses the same oracle files (sha `c9a07d9298a1`), JAX 0.10.2, and a v6e-1. Step 1 is static, so it runs
on CPU with an identical verdict. Some kernels halt the TPU core and make every later item in the same process fail; our harness
re-runs those victims in a fresh process and never counts them as verdicts.

### 3.3 Problems and contamination control

Problems come from KernelBench Level 1-2 and GPU MODE's KernelBook (both ported to JAX and checked against the PyTorch module),
operators written for this project (attention variants, MoE routing, quantized matmuls, KV-cache updates, SSM scans), and
references for public Pallas kernels. Every problem first passes seven CPU checks at full size (loads, fits HBM, deterministic and
seed-varying inputs, finite outputs).

**The exam** is all 73 JAXBench files (50 level 1 references, 23 level 2 kernels). A training problem is excluded if it is the same
kernel as an exam instance:
- **Copies:** KernelBench stems that JAXBench ported, and near-duplicate text on two token streams (identifiers kept, and identifiers
  abstracted).
- **Same op chain:** we trace `workload` to a jaxpr and canonicalize the primitive chain. A problem is out if its chain equals a
  k-task's once add, subtract, multiply, divide, negate and layout ops are set aside, and log-softmax is read as softmax. This is
  deliberately stricter than the rule it implements ("constants, scales and bias set aside"), and unsure means out.
- **Adversarial review:** a review of 273 problems that an earlier, family-level rule had excluded re-admitted 191 and kept out 43
  instance copies and 39 unsure cases.

Applied to the final data, the same-chain test removed:

| Source | Removed |
|---|---|
| Base problems | 58 |
| Scaled or teacher-added problems | 27 |
| Generated rows | 36 |
| Layout items | 20 |
| Softmax-template items | 2 |

The 29 dev items share no normalized op chain with any training row. No JAXBench task name appears in the data, and none of the 82
problems the review kept out is in it. Three independently written checkers verify all of this on the final file.

### 3.4 Training data (v7)

Only kernels that passed the oracle (`pass`, or `correct_slow` in the base set) become targets, and teacher reasoning is never
trained on. Rows are built by the evaluation's own prompt function, so they are byte-identical to what the model sees at test time.
A repair row is the problem, the failed kernel(s), the oracle's verbatim feedback and the passing kernel. Loss is on the final turn
only, so failed kernels are context, never targets.

**Teacher.** Kimi K3.1 (open weights, via Amazon Bedrock) wrote kernels for 1,672 KernelBench, KernelBook and authored problems with
a fixed prompt (kernel rules plus about 25k tokens of Pallas docs). On the first attempt 1,345 passed step 1 on CPU; of the
attempts judged on the TPU, 434 passed and 267 were correct but slow. Two repair rounds, each prompted with the earlier kernels
and the oracle's feedback, added 235 and 103 passing problems.

**Exam-scale rows.** Our earlier adapter (v4) failed mostly on VMEM: its matmul kernels loaded whole weights into VMEM, which fits at
training sizes and fails at JAXBench's (median weight 268 MB at K = 8,192, against 2 MB in our data). We added rows at those sizes in
three ways:
- **Scaling.** We scaled 198 matmul-dominant problems so that max(K, N) is between 4,096 and 16,384 and the weight is at least
  64 MB, then re-ran their existing kernels. 50 still passed (45 after contamination drops).
- **Fixing.** Kimi fixed the rest with the oracle's feedback and a TPU-measured tiling policy in the system prompt:
  - set `vmem_limit_bytes` to 124 MiB;
  - tile (tm, tn, tk) as (2048, 2048, 1024);
  - for row reductions, keep N whole with tm = 1024 and tk = 512 (256 at N = 8,192).

  After contamination drops, 48 problems with a passing first-round fix and 38 with a passing second-round fix remain. Each
  becomes a repair row and a turn-0 row.
- **Template generator.** A generator emits matmul-epilogue, batched-matmul and row-reduction kernels from one policy template.
  Each is checked numerically at a shrunken size, then judged at full size on the TPU:
  - 269 of 281 passed;
  - a layout variant, with transposed weights, bias before a reduction, and scale and shift, gave 104 of 111;
  - a whole-row output template for f32 softmax-type reductions at N = 8,192 gave 61 of 61.

  Almost every generated row (354 of 355) also appears once more in a **surface variant**: renamed arguments, CONFIG keys and docstring, so the policy is
  tied to the operation rather than the generator's wording. A variant is admitted only with two proofs: its target's AST equals
  the judged kernel's after renaming, and its reference is bit-identical at a shrunken size. A stratified sample of 91 variants then
  passed the TPU oracle 91 of 91.

**v4 gap.** Round 3 gave a third attempt to 486 problems that the earlier dataset covered but Kimi had failed twice. 103 passed.

**Composition.**

| Part | Rows |
|---|---|
| Kimi base set (after contamination and slow-matmul drops) | 1,172 |
| Exam-scale Kimi kernels (unchanged, fix rounds 1-2) | 217 |
| Round 3 (v4 gap; repair + turn-0 rows) | 206 |
| Template rows (originals + surface variants) | 709 |
| **Total** | **2,304** |

Teachers: Kimi 1,595 rows, generator 709 (30.8%). Kinds:
- 1,512 single-turn rows;
- 444 repairs;
- 348 rows whose target is correct but below 0.7x (15%). All are non-matmul base rows: below-bar matmul rows were dropped, since they
  teach exactly the untiled designs v7 must unlearn.

The longest row is 11,854 tokens, and the longest answer 5,819 (the eval cap is 8,192). Speeds have heavy outliers (38 rows above 5x),
so we report medians.

### 3.5 Training and serving

LoRA rank 64, alpha 128, on the attention and MLP projections of Gemma 4 E4B:
- 2 epochs, learning rate 1e-4, gradient accumulation 4, loss on the final assistant turn;
- 1,152 optimizer steps (AdamW, cosine schedule) in 89 minutes on one A100-80GB;
- mean training loss 0.157 in epoch 1 and 0.075 in epoch 2.

We score the final adapter; the epoch was fixed before training.

**The fold matters.** We serve with Ollama, which does not load LoRA adapters, so the adapter is folded into the official 4-bit QAT
GGUF that the baselines use. Our first adapter was trained on bf16 weights and folded back into Q4_0. Its median weight change was
2% of one Q4_0 rounding step, so rounding erased it: val kernels using `ShapeDtypeStruct` fell from 19 of 23 (bf16) to 6 of 23 as
served. Since v4 we:
- **train against the exact dequantized weights Ollama serves**;
- **store the 258 LoRA-touched tensors at Q8_0** as `dequant(Q4_0) + delta`, with every other tensor byte-identical.

The trainer's and Ollama's prompt token counts agree on all 64 turn-0 prompts.

### 3.6 Evaluation

**Tasks.** JAXBench's 50 level 1 tasks and 14 level 2-only tasks with our plain-JAX references, checked on the TPU against Google's
kernels. The **primary set** is the 58 whose references verified: 33 k-tasks and 25 p-tasks.

**Budget.** Every arm gets a first attempt and two repairs with the oracle's feedback, 3 attempts per task (against up to 144 in the
JAXBench paper). Gemma arms decode greedily (one chain, thinking off, 8,192 output tokens) on Modal A10Gs with Ollama 0.40.14.

**Metrics.**
- `pass_within_3`: a pass on any attempt.
- `correct_within_3`.
- `pass@1`.
- Parity: correct at 0.97x or faster.

Paired comparisons use an exact two-sided McNemar test on per-task outcomes.

**Arms.**
- Base Gemma 4 E4B with the reference only (`bare`), and with 25k tokens of Pallas docs (`docs`).
- Gemini Flash with docs, at its default sampling.
- Every LoRA we evaluated, in order: a 25-row pilot, v1, v4, v5 and v7.

No task, reference or kernel from JAXBench is used for training or model selection. JAXBench is run once per adapter, and every run
is reported.

## 4. Results

### 4.1 JAXBench

| Arm | Pass within 3 | Correct within 3 | pass@1 | Parity | k-33 | p-25 |
|---|---|---|---|---|---|---|
| Gemma 4 E4B, bare | 0/58 | 0 | 0 | 0 | 0 | 0 |
| Gemma 4 E4B + docs | 0/58 | 0 | 0 | 0 | 0 | 0 |
| Gemini Flash + docs | 7/58 | 36 | 1 | 4 | 5 | 2 |
| LoRA pilot (25 rows) | 0/58 | 0 | 0 | 0 | 0 | 0 |
| LoRA v1 | 0/58 | 0 | 0 | 0 | 0 | 0 |
| LoRA v4 | 1/58 | 3 | 1 | 1 | 0 | 1 |
| LoRA v5 | 1/58 | 2 | 0 | 1 | 0 | 1 |
| **LoRA v7 (final)** | **9/58** | **9** | **7** | **9** | **8** | **1** |

v7 passes:
- 12p (RMSNorm) and 19k, 20k, 27k, 37k, 41k, 44k on the first attempt;
- 35k and 47k after one repair.

Every pass runs at 1.30-1.39x XLA.

Paired comparisons with v7 (pass within 3):

| v7 vs | v7 only | Other only | p |
|---|---|---|---|
| Base Gemma (bare or docs) | 9 | 0 | 0.004 |
| v4 | 8 | 0 | 0.008 |
| v5 | 8 | 0 | 0.008 |
| Gemini Flash + docs | 8 | 6 | 0.79 |

v7 and Flash solve almost disjoint sets: only 12p is common. Flash is far more often *correct* (36 tasks), but its kernels are slow:
its best correct kernel per task has a median of 0.54x XLA (0.31x on first attempts), against 1.37x for v7. v7 is rarely correct
without being fast.

### 4.2 What changed between v4 and v7

**VMEM.** v4 had 16 VMEM failures at turn 0 and kept all 16 through three turns. v7 has 8 at turn 0. On repair, 2 of those became
passes and 6 now fit in VMEM but compute the wrong values. All 6 are matmuls followed by a row reduction (logsumexp, softmax,
sigmoid-sum, row max), whose epilogue must see a whole row while K is tiled.

**Compile errors are not repaired.** 41 of v7's 64 first attempts fail to compile, as many as v4's. In 80 of the 81 cases where a
compile failure was followed by another attempt, the next attempt also failed to compile. The oracle's feedback fixes memory layout,
not API misuse.

**p-tasks are unchanged.** Only RMSNorm passes, as it did for v4. The other p-tasks are production kernels (paged and ragged
attention, MoE, SSMs), and v7's added data is matmul-centric.

### 4.3 The path to v7

| Adapter | Data | Result | Lesson |
|---|---|---|---|
| v1 | 264 rows | 0/64 | The adapter was erased by a bf16-trained, Q4_0-folded serve |
| v4 | 2,983 rows; exact-Q4 training + Q8_0 fold | 1/58 | Learned the format, but matmuls died in VMEM at exam sizes |
| v5 | v4 with a plan before the code | 1/58 | Same failure |
| **v7** | **Kimi-only targets, exam-scale rows, stricter contamination rule** | **9/58** | — |

v1 and v4 used Claude-written targets; v7 uses only kernels from an open-weights teacher and our generator.

## 5. Discussion and limitations

- **JAXBench informed v7's design.** The exam-scale data exists because v4 failed on VMEM at JAXBench's sizes. No JAXBench task,
  reference or kernel is in training, and the same-chain test removed every row whose chain matches a k-task (including the chains v7
  passes). But v7 is the fifth adapter we evaluated, chosen as the last iteration rather than on a held-out benchmark, and it is
  tuned to a failure mode this benchmark revealed.
- **Testing is not verification.** Reading the targets after training, we found 5 rows (0.2%) whose kernels pass the oracle through
  input-dependent shortcuts:
  - a mean in place of a median, at 3,266x XLA;
  - polynomial approximations accurate only on the reference's input range.

  Three seeds from fixed input distributions cannot rule these out. v7 trained on them; the published dataset excludes them.
- **Template share.** 30.8% of rows come from one generator template. Over two epochs, each template kernel (with its variant) and
  each fix or round-3 target (repair + turn-0 row) is seen 4 times, other teacher rows twice. The dev set is template items, so its loss, which fell to 0.0002, measures template recall, not
  tiling ability. JAXBench is our only behavioural evidence.
- **Slow targets.** 15% of targets are correct but below the bar, and most Kimi kernels set no VMEM limit, while the fix and
  generator rows set 124 MiB. The model sees mixed signals outside matmuls.
- **Decoding and budget.** Gemma decodes greedily with one chain; Flash samples at its defaults because it loops at temperature 0.
  Three attempts is far below JAXBench's 144, and one greedy chain gives no confidence interval per task.
- **Timing.** We time with the host clock around `block_until_ready`. Both sides of the ratio pay host overhead, which pulls tiny
  kernels toward 1.0x.
- **Invalid runs.** Two early local runs served the base model under an adapter's label. We caught this by checking digests and
  excluded those runs; every reported row carries the served model's digest.

## 6. Conclusion

A 4B model can learn to write fast Pallas kernels for the problems its data covers. Teaching v7 to tile matmuls at exam size took
JAXBench passes from 1 to 9 of 58, on par with Gemini Flash with docs, from 2,304 verified rows and two epochs of LoRA. What remains
is API correctness (most failures still never compile) and production kernels outside the matmul family.

## Reproducibility

Repository `Art4v/gemma-hackathon-research`:
- **Oracle:** `oracle/` (sha `c9a07d9298a1`).
- **Final data:**
  - what v7 trained on: `hruday/kimi-k31:kimi/v7/v7_final.jsonl` (sha256 `10345461…`), with the val file, the build and check
    scripts, and the A6 review evidence;
  - the published release: `v7_release.jsonl`, which is the same minus the 5 shortcut rows (2,299 rows, sha256 `002da96d…`).
- **Evaluation:** `control/bench.py`, and `hruday/v4-runs:control/scoring/` for the fold and the JAXBench runner.
- **Results:** every run in `paper/data/runs/`, plus `paper/analysis/jaxbench.py`, which recomputes every table here. Oracle verdicts
  are on the `oracle-results/*` branches.
