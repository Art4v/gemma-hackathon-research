# Fine-tuning Gemma 4 E4B for JAXBench: from-scratch plan, audit of our runs, and what to do

Written Nov 11 2026, about 17:45 UTC, by Karthik's agent at Karthik's request. Karthik asked for
the fine-tune to be designed from scratch, without reading any of our plans, and then compared
against what we ran. Sections 1 and 2 were fixed before I opened the repo or the issues. Sections
3 to 6 are the comparison.

Nothing here was run on a TPU. Counts marked "regex" are approximate. No JAXBench output is used
as training material anywhere in this document; exam files are cited only to describe the exam.

## Summary

- The claim that our training tasks are at a different scale from JAXBench is right, and it is the
  dominant cause on the tasks we can win. It is more specific than "scale": the weight dimensions
  K and N are small in training, so the design that is correct in training is wrong on the exam.
- v4 learned the Pallas API. Structural validity went from 0 to 23 of 58 tasks, against 6 for base
  plus docs. It is two design rules short on the 25 matmul-led k-tasks, and the training data
  teaches the opposite of both.
- We do not lack rows. We lack exam-shaped rows. Going from 238 rows (v1) to 2,983 (v4) moved
  structural validity from 0 to 23 and passes from 0 to 1.
- The recipe, the prompt format, the quantized serving path and the verification are sound and are
  not the cause.
- For the time left: v7 should be weight-scaled matmul rows under one tiling policy, built by
  re-running verified kernels at scale before paying a teacher, with a held-out scaled dev set.

## 1. What the prior work says

GPU kernel fine-tunes (PyTorch to Triton or CUDA):

| Work | Base | Data | Method | Result on KernelBench |
|---|---|---|---|---|
| KernelLLM | Llama 3.1 8B | about 25k pairs, Triton side from `torch.compile` (KernelBook) | full SFT, 10 epochs | L1 pass@1 20.2, pass@10 51.8 |
| AutoTriton | Seed-Coder 8B | 14,102 SFT samples, teacher distillation plus compiler output, test-filtered | SFT then GRPO | L1 exec 10 (base), 29 (SFT), 36 (SFT+RL) |
| TritonRL | Qwen3 8B | 11k KernelBook tasks, up to 10 teacher samples each | SFT then GRPO | L1 pass@10 correct 23 (base), 54-64 (SFT), 78-88 (SFT+RL) |
| ConCuR / KernelCoder | QwQ 32B | 4,892 rows curated from 90,810 candidates | LoRA r32, lr 1e-4, 3 epochs | L1 pass@1 58 |
| Kevin | QwQ 32B | 180 KernelBench tasks | multi-turn GRPO, no SFT | 65% average correctness |
| Dr. Kernel | Qwen3 8B, 14B | 8k multi-turn trajectories | SFT then multi-turn RL | L2 Fast@1.2 5.6 (SFT), 20.0 (RL) |

Some of these numbers reached me through summaries of the papers, so check the source before
quoting one in the writeup.

Lessons that recur across at least two of them:

1. SFT is the foundation. Small bases score near zero and SFT gives a 2x to 3x gain. RL adds 10 to
   25 points on top, takes days, and collapses easily on small models.
2. Every training target is filtered by execution against the reference.
3. Curation beats volume. ConCuR's 4,892 curated rows beat a random 5k by 19 points. Its size
   ablation (pass@10, L1): 552 rows 72, 1,105 rows 80, 2,210 rows 86, 4,892 rows 91.
4. Several input shapes per task help. TritonRL's input augmentation moved 76 to 88.
5. Short reasoning beats long reasoning (ConCuR).
6. Small models learn to delegate to the framework. A syntax check is not enough; the check has to
   confirm the kernel does the work. Our outside-op rule already does this.
7. Train and eval distributions matched in all of these. KernelBook and KernelBench are both small
   PyTorch modules.

The exam (JAXBench, arXiv 2607.20466):

- 50 level-1 workloads on TPU v6e: 17 production LLM operators and 33 KernelBench L2 fusions
  resized to saturate the MXU, for example `(4096, 8192) x (8192, 8192)`.
- Of the 33 k-tasks, 25 are Matmul/Gemm/BMM plus an elementwise or reduction epilogue and 8 are
  conv-led.
- Gemini 3 Flash without docs: 13 of 50 tasks correct with 144 samples each, and 99.7% of one-shot
  samples fail at compile or first run on hallucinated APIs. Adding docs raises per-sample
  correctness from 5.8% to 37.3%. The authors conclude the bottleneck is information, not
  reasoning, and name fine-tuning as the next step.
- MultiKernelBench found the same on Pallas: best Pass@1 8.4%, top runtime error "unexpected
  keyword argument" at 22.4%.

One difference from the GPU work matters more than any other. On a GPU, a kernel that is correct at
a small size is correct at a large size. On TPU, the design changes once an operand no longer fits
in VMEM (128 MiB physical on v6e, 32 MiB default scoped limit). That is why input scale never
appears as a problem in the Triton and CUDA papers and is the first problem here.

## 2. The plan I would build from scratch

1. **Start from the exam's shape.** Generate training problems in the exam's families and at the
   exam's sizes, with exam op chains filtered out. Do not import the GPU recipe's problem pool.
2. **Get targets from skeletons, not free-form teacher output.** There is no JAX-to-Pallas compiler
   to play the role `torch.compile` plays for Triton. The substitute is one verified tiled-matmul
   skeleton plus a registry of epilogue snippets that are known to lower, composed
   programmatically. Yield is near 100% and every target has the same design. Teachers are for the
   long tail (attention, MoE, sequence operators).
3. **Put the arithmetic in code.** Every target picks its tiles through one helper and one VMEM
   limit, from shapes at trace time. A 4B model copies patterns; it does not do byte budgets.
   The same code is then valid at every size, which removes scale as something the model has to
   reason about.
4. **Verify at exam size on the chip.** A CPU lowering check does not see VMEM.
5. **Train only on kernels that meet the speed bar**, in families where the bar is reachable, since
   the bar defines a pass.
6. **Keep the prompt and output format byte-identical to the eval, loss on the answer only, no
   truncation.**
7. **Select on a dev set shaped like the exam**, never on the exam.
8. **Then expert iteration**: sample the student on training problems it has not seen, verify, add
   its own verified successes and its own failure-plus-feedback-plus-fix chains, retrain. Full RL
   only after that.

## 3. What we ran, against that plan

What matches, and is not the cause of the low score:

- Prompts byte-identical between training and eval; loss on the final answer; no truncation.
- TPU-verified targets on 3 seeds with an outside-op rule.
- Contamination filters stricter than anything in the literature.
- LoRA r64 on attention and MLP, lr 1e-4, 2 epochs; quantization-aware fold; digest checks.

Where it diverges:

**3.1 The problem pool is KernelBook-shaped, not JAXBench-shaped.** 1,522 of 2,983 v4 rows (51%)
and 1,101 of 1,332 v6 rows (83%) come from KernelBook, which is small scraped GitHub modules.
JAXBench's primary set is 25 matmul-epilogue tasks at 8192-wide weights, 8 conv tasks and 25
production operators.

**3.2 Two design rules are missing on the matmul k-tasks.** v4final-e2, turn 0, the 25
Matmul/Gemm/BMM-led k-tasks (my count from `bare.jsonl` and `verdicts.jsonl` on `hruday/v4-runs`):

| Design v4 chose | Tasks | vmem | wrong | compile | correct_slow | pass |
|---|---|---|---|---|---|---|
| Whole K in one block (1-D or 2-D grid) | 16 | 11 | 0 | 4 | 1 | 0 |
| K-accumulating (scratch plus `program_id`) | 9 | 2 | 4 | 3 | 0 | 0 |

- Rule A: tile K when the weight does not fit. v4 picks whole-K on 16 of 25 because that is the
  right answer in training. Hruday's measurement: median K 1,023 in v6 matmul rows against 8,192
  on the k-tasks; weight above 64 MB in 11% of rows against 86%. Targets with a 3-D or higher
  grid: 160 of 1,719 matmul targets in v4 (9%), 170 of 641 in v6 (27%) (regex on `grid=`).
  The e3 scale variants grew the batch, not the weights.
- Rule B: keep a reduced axis whole in the block. All 4 `wrong` verdicts (21k, 24k, 36k, 42k) are
  this one bug. The epilogue takes `sum`, `max` or logsumexp over `axis=1` of a 512-wide N tile
  instead of the 8192-wide row. 36k and 42k also only ever read N tile 0
  (`lambda i, k: (k, 0)`). I read all four kernels.
- The 7 compile failures in this family are plumbing: a ref used without `[...]` (43k), `.astype`
  called on a BlockSpec (31k), a 2-D block for a 1-D bias (45k), a captured constant (35k), and
  shape mismatches around reductions and group norm (23k, 32k, 40k).

**3.3 The targets give the model no policy to copy.**

- v4 sets `vmem_limit_bytes` in 2,276 of 2,983 targets, with four values: 64 MiB in 1,011 targets,
  48 in 301, 100 in 349, 96 in 211. 19k, 37k, 47k, 49k and 30k died at 65 to 66 MB against their
  own 64 MiB on a chip with 128 MiB.
- Kimi sets it in 0 of 1,332 v6 targets. An exam-scale matmul whose blocks fit in the 32 MiB
  default is the 0.2x to 0.34x regime v4 hit on 8p_GEMM with 256/512 tiles.
- Literal tile sizes in matmul targets range from 16 to 4096.
- 586 v4 targets (20%) and 240 v6 targets (18%) choose a block with
  `X if n % X == 0 else <whole dim>`. At exam size the fallback silently becomes "whole dimension
  in VMEM". Aarav's agent counts 781 in v4 with a broader regex, so the share is 20 to 25%.
- The 999 `repair_injected` targets are the same verified kernels as `single` targets, so they
  double the weight of whatever tiling those targets carry (Aarav's agent).

**3.4 v6 trains on slow kernels.** 518 of 1,332 v6 targets (39%) are below the 0.7x bar that
defines `pass`, and 157 are below 0.3x. v4 had 47 of 2,983. Among K-accumulating matmul targets in
v6 the median is 0.67x and 52% are below the bar.

**3.5 Plans (v5) asked the model for arithmetic.** v5 learned the `# PLAN` format on 64 of 64
replies, and step 1 fell from 23 to 12 (p = 0.007). The plans state byte sizes as literals, about
5% of them were wrong in the data, and a 4B model does not compute them. The useful form of a plan
is executable: a helper that computes tiles from shapes.

**3.6 Repairs copy.** 999 of 1,240 repair rows are planted-bug undos, where the fix is nearly the
previous code. v4 re-emits byte-identical code on 39 of 63 turn-1 repairs. `pass_within_3` is in
practice pass at turn 0.

**3.7 The eval cannot separate adapters.** At 1 of 58, McNemar needs 6 discordant tasks. The
23-problem val is conv-heavy KernelBench at toy sizes and only ever passes 3 elementwise problems,
so it cannot show whether an adapter fixed VMEM. Each JAXBench run costs $6 to $7 and counts under
A5. There is no dev set shaped like the exam.

**3.8 Out of reach in the time left.** The 8 conv-led k-tasks are 0 of 8 at the trace stage, on
shape bookkeeping in long kernels. The production operators are 6 of 31 at step 1. A 4B base is
half the size of the smallest model in the GPU literature.

Causes I checked and rule out as primary: training-row count, prompt-format mismatch, truncation,
loss masking, LoRA hyperparameters, the Q4 fold, greedy loops (fixed by epoch 2).

## 4. What to do before the Nov 12 12:00 UTC cutoff

Ranked.

1. **v7 is weight-scaled matmul rows under one tiling policy.** This is Hruday's direction and I
   back it. Build the rows by reuse before paying a teacher:
   1. Take the solved, unflagged, matmul-heavy pool problems and raise K and N to 4096-8192.
   2. Run each problem's existing verified kernel, unchanged, at the scaled CONFIG. Targets read
      shapes at runtime, so this needs no generation. CPU step 1 and the block-bytes estimate
      first, then the TPU.
   3. A pass is a `single` row at exam scale for zero teacher tokens.
   4. A `vmem` verdict is a real repair chain: a kernel that was right at small size, the oracle's
      real feedback, then the fix. It is the same failure v4 shows on JAXBench, and it is the
      real-feedback repair we have 241 of against 999 injected. It is the teacher's failure, not
      the student's own output, so the writeup must not call it on-policy data (Aarav's agent).
   5. Only the failures go to a constructor, as a repair turn with that feedback.
2. **One policy across every new target.** The same helper, one `vmem_limit_bytes` (about
   100 MiB), tiles of 1024 or more where the dimension allows. Choose the tile sizes with a small
   sweep on a few scaled training problems on the TPU, not from any exam file.
3. **Include rule B rows.** Scaled problems whose epilogue reduces over N, with targets whose
   block spans all of N.
4. **Clean the base set.** Drop sub-bar matmul rows, at least the 157 below 0.3x. Decide v4 plus
   v6 against v6 alone; v6 alone loses the `vmem_limit_bytes` habit.
5. **Hold out 25 to 30 scaled problems as a dev set**, never trained. Hold out whole problems,
   every small and scaled variant, or the dev score partly measures recall of a memorised kernel
   (Aarav's agent). Score turn 0 only. It shows
   whether v7 fixed VMEM before the one JAXBench run, and it is a legal way to choose between v7
   variants.
6. **Stop** plan variants, self-sampling on the old pool, conv tasks and new production-operator
   data.

What to expect. This can turn many of the 15 vmem-or-wrong matmul tasks into correct ones. A pass
also needs 0.7x of XLA on compute-bound matmul. v4's best on 8p_GEMM is 0.34x; the JAXBench paper
reports the hand-tuned upstream GEMM at 0.96x with large tiles. I have not verified on a TPU that
the f32 k-tasks can reach the bar, so report `correct_within_3` and the funnel next to
`pass_within_3`.

For the writeup. The claim that holds today is stage (b): 23 of 58 against 0 for base and 6 for
base plus docs. The scale finding is itself a result: the GPU recipe's problem pool does not
transfer to TPU because VMEM makes the kernel design depend on operand size.

## 5. With more time

1. A skeleton-plus-epilogue generator for the matmul and conv families at exam size, with exam op
   chains filtered by our existing contamination layers.
2. The uniform tiling helper in every target, including a rewrite of the existing matmul targets.
3. On-policy expert iteration on unseen training problems, with real-feedback repair chains
   replacing the injected ones.
4. A dev set of 50 or more exam-shaped problems with TPU verdicts, used for every selection.
5. RL from the oracle only after SFT plateaus on that dev set.

## 6. Open items

- A6 (the instance-level contamination rule) needs Karthik's and Aarav's sign-off. I have not
  reviewed it.
- I offered on #5 to do the CPU side of step 4.1 (scaled CONFIGs, step 1, VMEM estimate) or to be
  the constructor for the retiled fixes. Neither is started.

## Sources

- JAXBench: https://arxiv.org/html/2607.20466 and
  https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/JAXBench
- MaxKernel: https://arxiv.org/html/2609.04523v1
- Kevin-32B: https://cognition.ai/blog/kevin-32b
- TritonRL: https://arxiv.org/html/2510.17891v2
- AutoTriton: https://arxiv.org/html/2507.05687v1
- Dr. Kernel: https://arxiv.org/html/2602.05885v2
- KernelLLM: https://huggingface.co/facebook/KernelLLM
- KernelBook: https://huggingface.co/datasets/GPUMODE/KernelBook
- ConCuR: https://arxiv.org/html/2510.07356v1
- MultiKernelBench: https://arxiv.org/html/2507.17773v2
- Pallas TPU details: https://docs.jax.dev/en/latest/pallas/tpu/details.html
- Our data: `hruday/v4-runs` (`control/runs/`), `aarav/dataset-v4` (`dataset/v4/final/train.jsonl`),
  `hruday/kimi-k31` (`kimi/v6/v6_train_strict.jsonl`), issue #5.
