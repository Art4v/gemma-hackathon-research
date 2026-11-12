# Proposed protocol amendment A6: instance-level contamination rule for the v6 training data

**Status:** proposed on Nov 11, 2026, ~10:00 UTC. Hruday decided to adopt the rule on Nov 11. Claude drafted this text overnight while
Hruday was asleep, so he has not reviewed the wording yet. It needs sign-off from Karthik and Aarav. It is written **after** the v4 and
v5 JAXBench results were seen. That timing is disclosed here and must be disclosed in the writeup.

**Scope:** which problems may appear in training data. Nothing about the evaluation changes: the 64 tasks, the primary set of 58,
S-p (25), the prompts, decoding, 3 turns and the oracle `c9a07d9298a1` all stay as they were.

## The rule until now (rule 3, family level)

The previous rule dropped any training problem that was in the same *algorithm family* as a JAXBench task. Its sources were:
- the rule-3 algorithm reviews: `algo_review_{kbk,kbkc,kbkd}.jsonl`, plus `prod_excluded_by_algorithm.jsonl`;
- the op-multiset (`filter_extra`), op-chain (L2) and text near-dup (L3) layers;
- the v4 `exclude_families.txt`.

Together these excluded 418 problems:

| Class | Count | What it is |
|---|---|---|
| A | 93 | Instance-level copies, for example the KernelBench L2 stems that JAXBench itself ported |
| B | 273 | Family-level only: 135 "near", 138 "far" |
| C | 52 | Protects the dataset's own val split |

## The amended rule (instance level)

A training problem is excluded if it is **the same kernel as a JAXBench task instance**.

1. **p-tasks (human-written Pallas kernels).** The options that a JAXBench human kernel exposes count as the *same* kernel:
   - causal masking
   - additive bias
   - logit softcap
   - sliding window
   - attention sinks
   - GQA grouping
   - page table
   - LSE output

   A training problem whose dominant computation (at least 2/3 of FLOPs, or of bytes for memory-bound ops) is that kernel under any
   of these options is excluded.
2. **k-tasks (KernelBench L2 chains).** Adding or removing a nonlinear op makes a *different* problem. Changing constants, scales or
   bias does not. A training problem is excluded if the chain covering at least 2/3 of its FLOPs equals a k-task's chain once those
   are set aside.
3. **Exact copies are excluded as before.** This covers the ported KernelBench stems, the P_TASK_TWINS list and text near-duplicates.
4. **Unsure means out.** A problem the review cannot place is excluded by default.
5. **The val split keeps its protection.** This means the exact val task name plus the val-family check (for example `__bf16` copies of val problems).

### How it was applied

An adversarial review of all 273 class-B problems judged each one against its closest JAXBench task, using FLOP and byte shares
computed from the reference's jaxpr. The per-problem files are `review/adv/adversarial_verdicts.jsonl` and `final_safe_list.txt`.

| Verdict | Count |
|---|---|
| Safe to re-admit | 191 |
| Keep excluded (instance) | 43 |
| Keep excluded (unsure, default) | 39 |

**164 of the 191 re-admits are used in v6 data:**

| Group | Count | What happened |
|---|---|---|
| Already in the pool | 42 | In the v6 draft from the start. 40 are used; 2 stay out because Aarav's layer-3 near-dup filter dropped them |
| Not in the pool, reference passes its CPU checks | 133 | Kimi K3.1 generated a kernel for each. Rows carry `set = "e1_readmit"` and are verified by the TPU oracle like every other v6 row |
| ...of which near-dups | 9 | Excluded by rule 3 (near-dups stay out) |
| ...of which used | 124 | |
| Not used | 16 | 15 have no authored reference; 1 failed its reference checks |

The adversary also scanned the whole existing pool with the jaxpr FLOP/byte scan. The scan found instance-level copies that the old
filters had missed, for example L2Norm/PixelNorm (12p) and GEGLU/shared-expert MLPs (9p), plus some "unsure" cases. All of them are
excluded in v6 (`v6/exclude_hard.txt`, `v6/flag_unsure.txt`). The findings are listed in `review/adv/pool_contamination.json`.

### Re-admitted problems by closest JAXBench task (all safe / used in v6 data)

| Task | Re-admitted (all safe / used) |
|---|---|
| 1p Flash Attention | 49 / 48 |
| 22k Conv2d_InstanceNorm_Divide | 21 / 21 |
| 13p Cross_Entropy | 16 / 15 |
| 38k Matmul_Dropout_Softmax | 16 / 15 |
| 9p SwiGLU_MLP | 13 / 12 |
| 5p Flex_Attention | 9 / 9 |
| 8p GEMM | 8 / 6 |
| 15p RetNet | 7 / 5 |
| 14p Ragged_Dot | 6 / 5 |
| 41k Gemm_Add_ReLU | 5 / 3 |
| 42k Gemm_Max_Subtract_GELU | 5 / 2 |
| 25k Conv3d_GroupNorm_Mean | 4 / 3 |
| 27k Matmul_Mish_Mish | 3 / 3 |
| 6p Paged_Attention | 3 / 3 |
| 56p DeepSeek_V4_MHC | 3 / 1 |
| 18k, 31k, 37k | 2 / 2 each |
| 30k | 2 / 1 |
| 2p, 21k, 34k, 36k, 39k, 45k | 1 / 1 each |
| 63p, 64p, 10p, 12p, 16p, 65p | none used |

"Used" means the problem is in v6's problem set. A row exists only where Kimi's kernel was TPU-verified correct.

## Decisions that go with the rule

- **Generic GEMM/conv + bias + ReLU stacks stay in** (Hruday's decision (c)). Each such problem has at least 2/3 of its FLOPs in
  layers like 41k/19k/18k but is a different problem by the k-task test. Its rows carry `generic_stack = true`. In the writeup, report
  8p, 18k, 19k, 20k and 41k as tasks with **in-family training neighbours**.
- **A known near-copy in v4's data: 12p.** v4's training data has 15 rows on 7 instance-level 12p (RMSNorm) copies
  (L2Norm/PixelNorm), plus 13 RMSNorm-named tasks. v4's single pass on 12p must be reported with that disclosure. These copies are
  excluded from v6.

## New v6 data drawn under this rule (Nov 11 night)

| Set | What it is |
|---|---|
| **e1** | The 133 re-admits above |
| **e2** | Karthik's kbkd batch (`karthik/kbk@036fe89`, 224 problems with status ok). It already passed the pipeline's contamination layers. After dedupe and reference checks, 221 remain; 32 of them are generic stacks (flagged) |
| **e3** | JAXBench-size *scale variants* of 296 solved, unflagged pool problems (details below) |

e3 in detail:
- **Size.** Inputs are 200-800 MB (median 403 MB; JAXBench k-tasks have a median of 384 MB, while training problems had a median of 128 MB).
- **How it was resized.** One integer CONFIG value was multiplied by an integer factor, almost always the batch. The op chain is unchanged.
- **Which problems.** A problem was eligible only if it carried no flag of any kind. Five variants whose resized FLOP mix crossed
  the stack or attention threshold were dropped.

## Until sign-off: a strict file alongside

Some v6 rows exist only because of this amendment: the e1 re-admits and the rows from v4's rule-3 families. They are listed in
`v6/a6_dependent_problem_ids.txt`.

| File | Contents |
|---|---|
| `v6/v6_train.jsonl` | Hruday's decision: the amended rule |
| `v6/v6_train_strict.jsonl` | The same rows **minus** the A6-dependent ones, so it meets the old rule |

Each file has its own sha256, so whichever the team signs off on can be trained.

## Why the timing matters, and how the writeup handles it

- The rule was set after the team had seen per-task JAXBench outcomes for v4 and v5. Knowing which tasks were solved could bias
  which problems feel "safe."
- **Mitigations.** The rule is written in terms of kernel identity, not results. Every verdict is published per problem, with its
  reason. "Unsure" defaults to out.
- **Reporting.** The writeup reports v6 per task, with a column marking tasks that have re-admitted or in-family training neighbours
  (the table above, plus 8p/18k/19k/20k/41k). Results are also given excluding those tasks.
