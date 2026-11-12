# Datasheet: `v7_final.jsonl` (dataset v7)

Written by Aarav's agent. The counts come from the file itself (`hruday/kimi-k31` @ `2efa6a5`); the licence statements come from the notices
recorded in the repo. **Anything marked "to confirm" has not been verified and needs a human decision before publication.**

## Summary

| | |
|---|---|
| File | `kimi/v7/v7_final.jsonl`, 2,304 rows, sha256 `10345461a48148d379f99c7d304180eff0d23bf9693568cbef99090db7abcda9` |
| Dev / validation | `kimi/v7/v7_final_val.jsonl`, 29 rows, sha256 `d09e86649892595c375769c7c779deeefa9fcaecf9a140f755b87b0166860fac`. It shares no task or normalised op chain with training. |
| Task | JAX reference function → Pallas TPU kernel (one Python module defining `workload`) that matches the reference numerically on TPU v6e |
| Format | Chat rows: `messages` (system, user, assistant[, user feedback, assistant …]), `thinking: false`, plus provenance fields. The prompts are byte-equal to the evaluation harness (`control/bench.py` `conversation()`). |
| Verification | Every final target was judged on a TPU v6e-1 by the team oracle (`c9a07d9298a1`: 3 input seeds, closeness to an f32 reference, speed vs XLA with a 0.7× bar), except surface variants, which inherit the verdict of their proven-equivalent original (91 of 91 sampled variants re-passed on TPU) |
| Trained model | v7 LoRA adapter, `gs://gemma-hackathon-adapters-5ec089ff/v7/final/adapter/` (sha256 `a1259bb6…2243`) |

## Composition

| `set` | Rows | What it is | Kernel author |
|---|---|---|---|
| `pool` | 821 | v6 base: problems ported to JAX from KernelBook (658), KernelBench (123), team-authored (36) and GitHub (4); Kimi's first attempts and repairs | Kimi K3.1 |
| `e1_readmit` | 49 | Problems re-admitted under amendment A6 (instance-level contamination rule) after an adversarial review | Kimi K3.1 |
| `e2_kbkd` | 103 | Karthik's kbkd batch of KernelBook problems | Kimi K3.1 |
| `e3_scale` | 199 | Pool problems with one CONFIG size multiplied to exam-like input sizes (200–800 MB) | Kimi K3.1 |
| `e4_kn` | 45 | Matmul-heavy pool problems with K and N raised to 4096–8192, solved by the existing kernel unchanged | Kimi K3.1 |
| `e4_fix`, `e4_fix2` | 96, 76 | Repair chains at exam scale: a kernel that failed (mostly VMEM), the oracle's real feedback, Kimi's verified fix; plus the fix as a turn-0 row | Kimi K3.1 |
| `r3_gap` | 206 | Third attempts on problems v4 had trained on but v6 lacked, with two rounds of real feedback: repair chain plus turn-0 row | Kimi K3.1 |
| `e5_gen` | 400 | Template-generated matmul / bmm / row-reduction kernels at exam scale (200 originals + 200 surface variants) | `v7_gen.py` |
| `e5_gen_layout` | 191 | Generator: weight stored (n, k), bias, scale-and-shift (96 + 95 variants) | `v7_gen.py --layout` |
| `e5_gen7r` | 118 | Generator: f32 whole-row reductions at N = 8192, accumulating in the output block (59 + 59 variants) | `v7_gen.py --rowout` |

- **By kind:** 1,512 `single`, 208 `correct_slow`, 444 `repair`, 140 `repair_slow`.
- **By length:** 1,720 one-turn rows, 465 two-attempt rows and 119 three-attempt rows.
- **By author:** Kimi K3.1 wrote 1,595 targets and the template generator 709 (30.8%). There are no Claude-written targets.
- **Speed:** the median `speed_vs_xla` is 0.97×.
  - **Below the bar:** 348 rows (15%) have a final target below 0.7×. These are the `*_slow` kinds, kept by decision for non-matmul problems. All matmul-led slow rows were removed.
  - **Outliers:** 38 rows are above 5× and 2 above 2,700×. Use medians, not means.

## Exclusions applied

- **Validation and dev:**
  - The 23 dataset-v1 val tasks and every variant (`__bf16`, `__scale`, …) are excluded by family and stem.
  - The 29 dev items are excluded by problem and by normalised op chain.
- **JAXBench contamination:** all problems were filtered against JAXBench's 50 level-1 references and 73 files:
  - L1: KernelBench stem;
  - L2: exact op chain;
  - L3: near-duplicate text (5-gram Jaccard ≥ 0.5 or difflib ratio ≥ 0.8);
  - A6 rule 2: the op chain equals a k-task's once constants, scales, bias and layout ops are set aside, with log_softmax read as softmax; unsure means out.

  This removed 57 base rows, 27 Kimi-added problems, 41 generated items, 19 layout items and 2 rowout items. Two agents independently checked that no row comes from the 82 problems the A6 review kept excluded.
- **Below-bar matmul:** 180 below-bar matmul-led base rows were removed (the target design v7 is meant to unlearn).
- **Duplicates and conflicts:** none. Three independent checkers confirm: no duplicate ids, no two different targets for one prompt, every history turn in the evaluation's form.

## Known issues

1. **Five targets pass the oracle without computing the reference in general.** Found by reading the code after training, not by the oracle. They should be removed from any published copy:
   - `kernelbook__IMQSteinKernel_5388.kimi.a1`: it replaces a median with a mean (3,266×);
   - `kernelbook__NAC_188__kn8.kimi.a2` and `…a2t0`: approximations valid "over the observed input range";
   - `kernelbook__PNTrainingSigmoid_7307__scale.kimi.a2`: cubic approximations valid on the reference inputs' [0, 1) range;
   - `kernelbook__CompositeActivation_1340.kimi.a2`: a polynomial valid only for small magnitudes.

   The oracle's 3-seed closeness test on fixed input distributions cannot rule out such input-dependent shortcuts.
2. **The VMEM-limit policy is not uniform.** 1,423 Kimi targets set no `vmem_limit_bytes` (32 MiB default), while all 709 generated targets and 96 Kimi fix targets set 124 MiB.
3. **Template share:** 30.8% of rows come from one generator, about half of them surface variants of the other half. Validation loss on the template-style dev set is therefore uninformative (it reached 0.0002).
4. **Fixed input distributions:** speed and correctness were measured at one input distribution per problem (uniform or normal draws from each reference's `create_inputs`).

## Licences and provenance (to confirm before any release)

| Source | Rows (by `source`) | Licence as recorded in the repo |
|---|---|---|
| KernelBook (GPU MODE) | 1,340 | Used under the **June 9 Researcher Reciprocity License** (`dataset/kernelbook/LICENSE.txt` on `karthik/kbk`). Each module also keeps its original repository's licence (`license_checked`, `repo_link`). **Attachment A applies to any model trained on rows derived from this data:** researchers must be able to generate with, evaluate and publish about it on the same terms as other users. That bears on how the v7 adapter is shared. |
| KernelBench | 168 | MIT (`dataset/PROCESS.md`) |
| Team-authored references | 75 | Team's own. **The repo has no licence yet: to confirm.** |
| GitHub-scraped (MaxText, tpu-inference and others) | 12 | Apache-2.0 for MaxText and tpu-inference (`dataset/PROCESS.md`); per-file `source_ref` records the rest. **To confirm per file.** |
| Generated by `v7_gen.py` | 709 | Team's own code. **Repo licence: to confirm.** |
| Kernel targets written by Kimi K3.1 (via AWS Bedrock) | 1,595 | **To confirm:** whether the model provider's terms allow publishing outputs as a training dataset. |
| JAXBench (evaluation only, not in the data) | 0 | Apache-2.0, unmodified copies in `benchmarks/jaxbench/` |

## Intended use

- **Intended:** research on fine-tuning small models to write TPU Pallas kernels; reproducing the paper's v7 result.
- **Not intended:**
  - as an evaluation set, since every problem is a training problem;
  - as a source of production kernels without re-verification on the target shapes and input distributions (see Known issues 1 and 4).
- **JAXBench is an exam only.** No JAXBench task, reference or kernel output is in this data.
