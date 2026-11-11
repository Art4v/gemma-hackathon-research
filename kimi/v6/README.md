# v6 training set (final, Nov 11 2026): Kimi K3.1 kernels the TPU oracle verified

| File | Rows | sha256 | What it is |
|---|---|---|---|
| `v6_train.jsonl` | **1,413** | `d9271a92ee265b0f2cac9002f6b5abc6f5ec0ce1dd9600f28fbc554f87a6a2b2` | Hruday's decision: the instance-level contamination rule, amendment A6 |
| `v6_train_strict.jsonl` | **1,332** | `9a7001c753f68ae7a898af804cfc6b27a5af82eab9c2a4f44760199f41b74570` | Drops the 81 rows whose admission depends on A6 (`a6_dependent_problem_ids.txt`: the e1 re-admits and v4's rule-3 families). Use it if A6 is not signed off |

## Format
- **Rows:** v4's exact row format, built with `control/bench.py` `conversation()`. The system prompt is identical to v4's. The user turn holds
  the reference.
- **Repairs:** rows add the failed kernel, then the oracle's feedback + "Reply with the corrected Python code block.", as JAXBench's repair turns do.
- **Targets:** code-only (`thinking: false`).
- **Problems:** 1,413 distinct problems, one row each. Every row keeps `problem_id`, so Kimi's reasoning and plan can be joined back from `kimi/data/`.
- **Tokens** (Gemma tokenizer, `train_lora` encoding):

  | | Median | p99 | Max | Over limit |
  |---|---|---|---|---|
  | Prompt + answer | 1,605 | 5,938 | 11,854 | none over 24,576 |
  | Answer | 788 | 2,869 | 5,819 | none over 8,192 |

## Composition
| Set | Rows | What it is |
|---|---|---|
| pool | 967 | The 1,672-problem Kimi pool (`data/full_problems.jsonl`), first attempts + repair round 2 |
| e1_readmit | 78 | Class-B problems re-admitted under A6 (`AMENDMENT_A6_instance_rule.md`) |
| e2_kbkd | 134 | Karthik's kbkd batch (`karthik/kbk@036fe89`, 221 problems after dedupe and reference checks) |
| e3_scale | 234 | JAXBench-size scale variants of solved, unflagged pool problems: inputs 200-800 MB, median 403 MB (`data/scale/`) |

| Kind | Rows | What it is |
|---|---|---|
| `single` | 591 | Attempt 1 passed on TPU |
| `correct_slow` | 357 | Correct but under the 0.7x speed bar |
| `repair` | 271 | Failed attempt + oracle feedback, then a verified fix (pass) |
| `repair_slow` | 194 | As `repair`, but the fix is correct but slow |

Flags kept on rows:
- `generic_stack` (122 rows): GEMM/conv+bias+ReLU stacks, kept by Hruday's decision (c). JAXBench 8p/18k/19k/20k/41k are reported as having in-family neighbours.
- `rule3_family` (3 rows).

## Exclusions (`exclude_hard.txt`, `flag_unsure.txt`)
- **Instance-level copies** of JAXBench tasks.
- **The adversarial review's "unsure" tier** (out by default).
- **All 29 problems from Aarav's filter pass on the draft.**
- **Val protection:** 15 bf16 copies of dataset-v1 val problems, plus a scale variant of one. 5 were found by Aarav's `family_id` check and 10 by stem match.
  `finalize_v6.py` refuses to write v6 if any row is a val problem or a copy of one.
- **9 re-admits** that the L3 text near-dup filter had caught (A6 rule 3).

## Verification (`scripts/finalize_v6.py`, plus an independent cross-check)
- **Verdicts:** every row's final kernel has a TPU verdict of `pass` or `correct_slow` (oracle `c9a07d9298a1`, TPU v6 lite, 3 seeds, speed bar 0.7) in
  `tpu/tpu_verdicts.jsonl`.
- **Code:** the target is byte-identical to the code the TPU judged.
- **Never used:** `harness_oom` / `too_large_for_chip` rows.

## Harness fixes made while building it (Nov 11; `tpu/crashloop.py`)
The oracle code is unchanged. All three fixes are in how its runs were driven and read.
1. **Crash suspects.** `tpu_oracle.py` judges items **sorted by task**, but crashloop blamed the next item in pack order. Suspects are now
   taken in the oracle's order.
2. **Halt cascades.** After one kernel halts the TPU core (its own verdict: `runtime`), every later item in that process got `harness_oom`
   ("program continuator has halted unexpectedly" / `RuntimeUnexpectedCoreHalt`). 233 of the 244 `harness_oom` rows were such
   victims. Their kernels are now re-run in a fresh process, and consumers ignore victim rows (`halt_victim` is flagged in
   `tpu_verdicts.jsonl`). The old-pool victims were re-judged (`kimi-k31-rej`, `kimi-a2rej`).
3. **Stranded rows.** An oracle process can judge every item, then hang at exit and be SIGTERMed. Its rows were left in `shards/` and never
   collected, so the part was re-run. `crashloop.sync()` now collects them, and 243 stranded rows were recovered.

The JAXBench eval jobs for v4final-e2 and v5 contain no halt cascades.

## Cost
| Item | Spend |
|---|---|
| Kimi K3.1 on Bedrock, new problems (e1 + e2 + e3, first attempts + repair round 2) | **$155.13** (cap $200) |
| Old-pool repair round 2, completed for the halt victims | $5.51 |
| First pool (earlier) | $346.45 first attempts + $113.05 repair round 2 |

All Kimi spend is on AWS credits. TPU time was on the GCP trial credit; every VM had its own $12 guard.
