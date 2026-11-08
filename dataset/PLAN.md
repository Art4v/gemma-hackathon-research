# Plan: plain-JAX problem set for the Pallas LoRA

Companion documents: `PROCESS.md` (how the set is built and checked, step by step) and `README.md` (handoff: what a record is).

## Context

Gemma 3n E4B is evaluated on JAXBench (65 tasks): it is shown a plain-JAX module and asked for a Pallas TPU kernel, scored by the
oracle in `oracle/`. Controls are E4B bare and E4B with Pallas docs; the test arm is E4B plus a LoRA.

On Nov 8 Pallas generation was paused. This dataset is now a set of **exam-shaped problems**: plain-JAX reference modules in exactly
the form JAXBench questions take, each verified on CPU to be ready for the oracle. It contains no Pallas code. Whoever trains the LoRA
produces the Pallas side (for example RL against the oracle, or a later generation run). The pilot set is the dataset; there is no
second, larger phase.

JAXBench is the final exam only. No JAXBench task, reference or human kernel appears here, and every problem is filtered against all
65 tasks.

## Decisions

| # | Decision |
|---|---|
| 1 | A record is a problem only: the reference module (`CONFIG`, `create_inputs()`, `workload`) plus metadata. No solution field. |
| 2 | The pilot set is the training set: KernelBench ports, constructor-authored model operators, and references for public Pallas kernels. |
| 3 | Every problem passes oracle-readiness on CPU at full size (`scripts/cpu_check.py`, seven checks, the oracle's own code). |
| 4 | Scraped references are checked against their public kernel in Pallas interpret mode on CPU; the kernel files are deleted afterwards. |
| 5 | Contamination: KernelBench-stem match, op-chain signature, and near-duplicate text against all 73 JAXBench baseline files. |
| 6 | Output: JAXBench-style folders (`final/problems/<task>/baseline.py`) and a JSONL index (`final/problems.jsonl`). |
| 7 | The paused Pallas pipeline is kept, unused, under `archive/pallas/`. |
| 8 | Memory: the build machine has 17 GB; heavy checks run one at a time under an RSS cap. |

Terminology: the model that writes references is the **constructor agent**, never "teacher" (used elsewhere in the study).

## Result (Nov 8)

| Source | Problems |
|---|---|
| KernelBench ports (torch-checked) | 19 |
| Constructor-authored model operators | 10 |
| References for public Pallas kernels (interpret-mode checked) | 3 |
| **Total** (30 train, 2 val by family hash) | **32** |

Excluded with reasons in `final/excluded.jsonl`: two KernelBench ports too close to exam tasks 42k and 44k, one port that failed the
PyTorch check (Frobenius norm, relative error 1.5e-3), one conv port over the memory cap even when shrunk, one scraped kernel whose
inputs were too small.

## Verification

- `scripts/cpu_check.py`: every kept problem passes all seven checks at full size (peak 6.7 GB, attention with sinks).
- `scripts/check_scraped.py`: all three scraped references agree with their kernel within the oracle's tolerance on 3 seeds.
- `scripts/export.py --check`: folder count equals records, every `baseline.py` sha matches its record, no family straddles the split,
  no `pallas_call` anywhere in `final/`.
- `python -m unittest discover -s dataset/tests -t .` passes.
