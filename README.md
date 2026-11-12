# Teaching a 4B Gemma to write TPU kernels

We fine-tune **Gemma 4 E4B** with LoRA on Pallas TPU kernels that a TPU oracle verified, and evaluate on **JAXBench** with three
attempts per task.

On the 58-task primary set, the adapter passes **9 tasks**, against 0 for base Gemma and 7 for Gemini Flash with 25k tokens of
Pallas docs, and 7 on the first attempt against Flash's 1. Every kernel that passes runs faster than XLA (1.30-1.39x).

| Arm | Pass within 3 (primary-58) | Correct within 3 | pass@1 |
|---|---|---|---|
| Gemma 4 E4B, bare | 0 | 0 | 0 |
| Gemma 4 E4B + docs | 0 | 0 | 0 |
| Gemini Flash + docs | 7 | 36 | 1 |
| **Gemma 4 E4B + our LoRA** | **9** | **9** | **7** |

- **Writeup:** [`paper/WRITEUP.md`](paper/WRITEUP.md). The gains are in production-scale tiled matmul kernels, built from
  training data at JAXBench's matmul scale; the limitations section covers how that data was built and where the kernels stop
  (partial tiles).
- **Companion notebook:** [`paper/notebook/submission.ipynb`](paper/notebook/submission.ipynb). CPU only, it recomputes every number
  from the published data and raw run files.
- **Data:** release [`dataset-final`](https://github.com/Art4v/gemma-hackathon-research/releases/tag/dataset-final).
  - `train.jsonl`: 2,299 rows, the published training set.
  - `train_as_trained.jsonl`: the 2,304 rows the adapter trained on.
  - `val.jsonl`.
  - A datasheet and checksums.

## Repository map

| Path | What it is |
|---|---|
| `oracle/` | The TPU oracle: kernel check, full-size run, 3-seed correctness, speed vs XLA. File hash `c9a07d9298a1` throughout. |
| `control/` | The JAXBench harness. `bench.py` holds prompts, generation and repair turns, and asserts the served model's digest on every row. `PROTOCOL.md` is the evaluation protocol, and `control/scoring/` holds the LoRA fold (`fold_local.py`) and the Modal JAXBench runner. |
| `kimi/` | The teacher pipeline: Kimi K3.1 generation and repair rounds (`scripts/`), TPU verdicts (`tpu/`), contamination lists and the adversarial-review evidence (`v6/`), and the final build and checks (`v7/`: `assemble_v7.py`, `check_v7.py`, rule-2 tracing in `a6r2/`, surface-variant proofs). |
| `training/` | The LoRA trainer (`train_lora_exact_q4.py`, trains against the exact 4-bit weights Ollama serves) and the final run's settings. |
| `dataset/` | The template generator (`v7_gen/`, `v7_gen_layout/`), its checker (`v7_check/`) and shared dataset scripts. |
| `paper/` | The writeup, figures, `analysis/jaxbench.py` (every JAXBench table), `data/runs/` (raw rows of every arm) and the notebook. |

## Reproduce the numbers

```
python3 paper/analysis/jaxbench.py        # writes paper/analysis/results.{json,md}; standard library only
```

Or run the notebook top to bottom. It downloads the release and the run rows, checks every sha256, and asserts the published counts.

## Provenance

- **Oracle verdicts:** every per-job verdict file is on an `oracle-results/*` branch.
- **Run rows:** every run row records its oracle hash, seeds, speed bar, chip, JAX version and the served model's digest.
- **Development history:** kept as `archive/*` tags, on `hruday/archive-local` (scripts and data that lived only on a laptop), and
  on the dataset and run branches of each team member.

## License

To be added by the authors. The third-party sources keep their own licenses: KernelBench is MIT, and KernelBook is its permissively
licensed subset. See the release datasheet.
