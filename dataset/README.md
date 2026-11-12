# dataset

> This file, `PLAN.md` and `PROCESS.md` describe the pilot problem set of Nov 8 (`final/`, `problems/`): 32 plain-JAX problems with no
> kernels. It is not the training data of the writeup. That data is the release
> [`dataset-final`](https://github.com/Art4v/gemma-hackathon-research/releases/tag/dataset-final) (2,304 rows), built by the teacher
> pipeline in `kimi/` and the template generator in `v7_gen/` and `v7_gen_layout/` here, with `v7_check/` as the generator's checker.

32 plain-JAX problems in exactly the form JAXBench questions take, each verified on CPU to be ready for the TPU oracle. This is the
training problem set for the Pallas LoRA. It contains no Pallas code: producing kernels for these problems is the trainer's job.

`PLAN.md` holds the decisions and results; `PROCESS.md` is how the set was built and checked.

## What is here

```
final/problems/<task_name>/baseline.py   the problem: CONFIG, create_inputs(), workload(*inputs), plain JAX
final/problems.jsonl                     one record per problem (fields below)
final/excluded.jsonl                     problems left out, with the reason
final/manifest.json                      counts, splits, sha256 of every file, JAX version, oracle sha
```

The folder layout matches `benchmarks/jaxbench/level1/<task>/baseline.py`, so any harness that reads JAXBench tasks reads these.
An oracle item for a solution is `{"id", "task": <task_name>, "ref_path": "final/problems/<task_name>/baseline.py", "code_path": ...}`.

## Record fields (`problems.jsonl`)

| Field | Meaning |
|---|---|
| `task_name`, `problem_id`, `family_id` | ids; `family_id` groups variants and drives the split |
| `source`, `source_ref` | `kernelbench`, `authored` or `scraped`; upstream repo, commit, path, licence |
| `path`, `ref_sha256` | the `baseline.py` file and its sha256 |
| `description` | the constructor agent's short note on what the operator is and the shapes chosen |
| `input_shapes`, `input_dtypes`, `input_bytes`, `output_shapes`, `output_dtypes` | at full size |
| `op_chain` | canonical primitive chain of `workload` (layout ops dropped) |
| `cpu_check` | the seven oracle-readiness checks, run time, peak memory, JAX version, oracle sha |
| `port_check` | KernelBench ports: max relative error vs the PyTorch module |
| `semantics_check` | scraped problems: max relative error vs the public kernel in Pallas interpret mode |
| `nearest_exam` | the closest JAXBench task by text and by op-chain, with similarity |
| `split` | `train` (30) or `val` (2) |

## Guarantees

- Every problem loads, fits a v6e (3 x input bytes <= 28 GB), builds deterministic inputs that change across the oracle's 3 seeds,
  runs at full size and gives finite outputs, including at the oracle's "highest" precision answer key.
- No JAXBench task, reference or kernel, and no KernelBench problem the exam uses, is in the set; every problem was filtered by name,
  op-chain and near-duplicate text against all 65 tasks.
- No problem family appears in both splits.

## Re-running

```bash
bash dataset/scripts/env.sh && source dataset/.venv/bin/activate
python dataset/scripts/cpu_check.py --force       # all problems, one at a time (peak ~7 GB)
python dataset/scripts/export.py && python dataset/scripts/export.py --check
SERA_JAX_PY=$PWD/dataset/.venv/bin/python python -m unittest discover -s dataset/tests -t .
```

`archive/pallas/` holds the paused kernel-generation pipeline; nothing in the dataset depends on it.
