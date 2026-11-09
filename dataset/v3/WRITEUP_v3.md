# Draft for the writeup: dataset v3

Numbers are from `aarav/dataset-v3` @ `2efe753` (`dataset/v3/final/MANIFEST.json`, `contamination_report.json`, `repairs_report.json`, `large_blocks_report.json`). Pending: `v3-gen-check50` and the capped-vs-full val ablation.

## Paragraph

Dataset-v1 has 238 training rows, and a second adapter trained on them with on-policy repairs showed memorization (validation loss rose from 0.35 to 0.47 as training loss fell to 0.07), so dataset v3 adds rows that need little or no new TPU judging. It has 2,877 training rows, and validation stays dataset-v1's 26 rows unchanged. New rows are derived only from training families.

**Injected repairs (1,167 rows).** Each of the 213 single-turn training kernels that the TPU oracle passed was broken in up to six ways that match the first adapter's errors: `out_shape` passed as an int, an index map of the wrong arity, the wrong number of `in_specs`, a block of the wrong rank, an invented keyword or API name, and a missing `workload`. All 1,167 mutants that could be built were rejected by the oracle's own CPU kernel check, and its feedback line is used verbatim as the repair turn. The target is the original TPU-verified kernel.

**Generated kernels (1,335 rows).** A template generator produced 2,400 reference and kernel pairs: matmul and batched matmul with random elementwise epilogues, row reductions, elementwise operations and reductions across ranks, and convolution via im2col. Each pair was checked on CPU with the oracle's kernel check, a real Mosaic lowering for a v6e chip, and interpret-mode numerics on three seeds against the oracle's answer key; 2,368 passed. A negative control of five deliberately wrong kernels was rejected five times out of five. These rows are CPU-verified only and are labelled as such.

**Extras.** 83 shape variants pair a dataset-v1 kernel, unchanged, with a reference whose configuration has a size entry halved (132 of 187 passed the CPU check; 83 survived filtering). 54 rows are kernels the TPU oracle found correct but below the 0.7× speed bar; they are labelled `correct_slow`.

**Contamination.** Every new reference was filtered at dataset-v1's thresholds against JAXBench (levels 1 and 2) and, for v3, against the 23 validation problems too. Of 2,500 CPU-verified generated and shape-variant references, 260 were removed: op-chain matches with JAXBench 12, op-multiset Jaccard ≥ 0.8 with JAXBench 74, op-chain matches with validation 95, op-multiset Jaccard ≥ 0.8 with validation 20, near-duplicate text 47, and within-set duplicates 12. No reference matched a JAXBench task on both op multiset and input shapes. Among kept rows the highest similarity is 0.794 by op multiset, and 0.499 Jaccard and 0.799 ratio by text. The generated pool was then cut to 1,500 by a stable hash.

**Chip check.** A random sample of 100 generated rows run on a v6e gave 89 pass, 10 VMEM and 1 `correct_slow`, with a median speed of 1.00× XLA for correct kernels. All failures were in the elementwise and reduction templates and were predictable from block size: below 8 MiB of input blocks per grid step, 30 of 30 passed, and at or above it 5 of 16 did. We removed the 165 rows at or above that limit, leaving 1,335.

**Limits.** The generated rows vary operations far more than API structure: by structural skeleton, the top 10 of 162 skeletons cover 83% of the kernels in the full training set, against 24% in dataset-v1. We therefore also provide a capped set of 1,597 rows with 55 generated rows, where the top 10 cover 30%, and choose between the two on validation. The 8 MiB rule was fitted on a 46-row sample. Most generated rows have not been on a chip, and CPU checks cannot see VMEM use or speed.

**Tokens** (Gemma 4 tokenizer and chat template): 6.31M per epoch for the full set and 5.19M for the capped set; longest row 14,839; longest target 5,920, within the evaluation's 8,192.

## Table

| Kind | Rows | Verification |
|---|---|---|
| dataset-v1 single | 213 | TPU `pass` |
| dataset-v1 repair | 25 | TPU `pass` |
| `repair_injected` | 1,167 | target TPU `pass`; failure and feedback from the CPU kernel check |
| `generated` | 1,335 (55 in capped) | CPU only |
| `shape_variant` | 83 | CPU only; target kernel was TPU `pass` at the original shape |
| `correct_slow` | 54 | TPU, correct but below the speed bar |
| **Train** | **2,877** (capped **1,597**) | |
| Val (dataset-v1) | 26 | TPU `pass` |

## Numbers to check before submission

- One level-2 reference could not be traced, `55p_DeepSeek_V4_StreamIndex_TopK`, so the op-chain filters did not cover it. The text filter did.
- The 260 removals cover generated and shape-variant references together (211 and 49). The 165 VMEM removals came afterwards and are separate.
- Update the chip-check figures after `v3-gen-check50`.
