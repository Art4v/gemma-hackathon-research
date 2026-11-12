# Teaching a 4B Gemma to write TPU kernels

A LoRA adapter that teaches Gemma 4 E4B to write JAX Pallas kernels for Google TPUs, with the TPU oracle that judged its training
data, the training set, and the JAXBench evaluation. This repository accompanies our Kaggle Gemma 4 writeup.

## Result

Pallas is JAX's kernel language for TPUs. Base Gemma 4 E4B passes none of JAXBench's 50 tasks, with or without 25k tokens of Pallas
documentation in its prompt. With our adapter and no documentation it passes 9 within three attempts, 7 of them on the first
attempt. Gemini Flash with the documentation passes 7, and 1 on the first attempt.

| Arm | Pass within 3 (JAXBench 50) | Correct within 3 | Pass at 1 |
|---|---|---|---|
| Gemma 4 E4B, bare | 0 | 0 | 0 |
| Gemma 4 E4B + docs | 0 | 0 | 0 |
| Gemini Flash + docs | 7 | 36 | 1 |
| **Gemma 4 E4B + our LoRA** | **9** | **9** | **7** |

A pass is a genuine Pallas kernel that compiles and runs at full size on a TPU v6e, matches a float32 reference on three seeds, and
reaches at least 0.7x the speed of XLA. "Correct" drops the speed requirement. Each task gets a first attempt and up to two repairs
with the oracle's feedback.

How to read the table:

- All nine passing kernels run faster than XLA (1.30x to 1.39x). Eight are tiled matmul kernels and the ninth is RMSNorm.
- Against base Gemma the paired split is 9 tasks to 0 (exact McNemar p = 0.004). Against Flash it is 8 to 6 (p = 0.79), so the two
  cannot be told apart on passes.
- Flash writes a correct kernel for 36 tasks, but most are slower than the bar. It is a larger model, run with thinking on and its
  default sampling, so it is a reference point and not a controlled comparison.
- The matmul training rows were sized to JAXBench's matmul scale, and the adapter's kernels assume sizes that their tiles divide.
  No JAXBench task, reference or kernel is in the training data. The writeup's limitations section has the details.

## What is published

- Writeup: [`paper/WRITEUP.md`](paper/WRITEUP.md).
- Companion notebook: [`paper/notebook/submission.ipynb`](paper/notebook/submission.ipynb). CPU only. It downloads the dataset
  release and the raw run rows at a pinned commit, checks the release files against their sha256, and recomputes the result table
  and the paired tests.
- Model: release [`model-final`](https://github.com/Art4v/gemma-hackathon-research/releases/tag/model-final). It holds the adapter
  (rank 64, 558 MB), a model card and checksums.
- Data: release [`dataset-final`](https://github.com/Art4v/gemma-hackathon-research/releases/tag/dataset-final).
  - `train.jsonl`: 2,299 rows, the published training set.
  - `train_as_trained.jsonl`: the 2,304 rows the adapter trained on. The 5 extra rows pass the oracle through input-dependent
    shortcuts and are listed in `excluded_rows.json`.
  - `val.jsonl`: 29 development rows.
  - A datasheet and checksums.

## Reproduce the numbers

The analysis needs only Python 3 and its standard library:

```
git clone https://github.com/Art4v/gemma-hackathon-research
cd gemma-hackathon-research
python3 paper/analysis/jaxbench.py
```

It reads the raw generations and verdicts of every arm in `paper/data/runs/`, rewrites `paper/analysis/results.json` and
`results.md`, and prints the tables. The first one is:

```
| Arm | pass within 3 (JAXBench 50) | correct within 3 | pass@1 | parity within 3 | k-33 pass | p-17 pass | extended-58 pass | all-64 pass |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 E4B (base), reference only | 0/50 | 0/50 | 0/50 | 0/50 | 0/33 | 0/17 | 0/58 | 0/64 |
| Gemma 4 E4B (base) + 25k tokens of Pallas docs | 0/50 | 0/50 | 0/50 | 0/50 | 0/33 | 0/17 | 0/58 | 0/64 |
| Gemini Flash + docs | 7/50 | 36/50 | 1/50 | 4/50 | 5/33 | 2/17 | 7/58 | 8/64 |
| Gemma 4 E4B + our LoRA | 9/50 | 9/50 | 7/50 | 9/50 | 8/33 | 1/17 | 9/58 | 9/64 |
```

The extended 58 adds the 8 level 2 tasks whose plain-JAX reference verified against Google's kernel on the TPU, and the 64 adds the
6 whose reference did not. The notebook recomputes the same table from GitHub and needs `matplotlib` for its one plot.

Judging a new kernel needs a TPU v6e. `oracle/DESCRIPTION.md` describes the oracle and `oracle/QUEUE.md` how to run a job on a VM.

## Use the adapter

The adapter was trained against the dequantized 4-bit weights of the official `gemma4:e4b-it-qat` GGUF, so it is served by folding
it into that file:

```
python control/scoring/fold_local.py <gemma4:e4b-it-qat model blob> <adapter directory> gemma4-e4b-pallas.gguf --dtype q8_0
```

This adds the LoRA update to the 258 adapted tensors, stores them at 8 bits, and copies every other tensor unchanged. The model blob
is the 5.15 GB file `sha256-e8b6a059...` under `~/.ollama/models/blobs/` after `ollama pull gemma4:e4b-it-qat`. The script needs
`numpy`, `gguf` and `modal` installed. Run on the released adapter, it reproduces the GGUF we evaluated byte for byte (sha256
`997a7145c280863d512dab29da7c961372e2111103171b2bf88d7d6a00391760`, 7.1 GB). The model card in the release gives the prompt and decoding settings, and
`control/scoring/README.md` the remaining steps we used to register and score the model.

## Tests

```
pip install -r requirements.txt          # JAX 0.10.2 on CPU; we used Python 3.12
python -m unittest discover -s oracle/tests
python -m unittest discover -s control/tests
python -m unittest discover -s dataset/tests -t .
python -m unittest discover -s curation/tests
python -m unittest discover -s pallas_guides/tests
```

That is 154 tests: 104 for the oracle, 9 for the benchmark harness, 23 for the dataset scripts, 15 for the CPU pre-filter (one
skipped) and 3 for the documentation build.

## Repository map

| Path | What it is |
|---|---|
| `oracle/` | The TPU oracle: kernel check, full-size run, 3-seed correctness, speed vs XLA. File hash `c9a07d9298a1` throughout. |
| `control/` | The JAXBench harness. `bench.py` holds the prompts, generation and repair turns, and asserts the served model's digest on every row. `PROTOCOL.md` is the evaluation protocol. `refs/` holds our plain-JAX references for the level 2 tasks, and `scoring/` the LoRA fold (`fold_local.py`) and the Modal JAXBench runner. |
| `kimi/` | The teacher pipeline: Kimi K3.1 generation and repair rounds (`scripts/`), TPU verdicts (`tpu/`), contamination lists and the adversarial-review evidence (`v6/`), and the final build and checks (`v7/`: `assemble_v7.py`, `check_v7.py`, rule-2 tracing in `a6r2/`, surface-variant proofs). |
| `training/` | The LoRA trainer (`train_lora_exact_q4.py`, which trains against the exact 4-bit weights Ollama serves) and the final run's settings. |
| `dataset/` | The template generator (`v7_gen/`, `v7_gen_layout/`), its checker (`v7_check/`) and shared dataset scripts. `final/`, `problems/`, `PLAN.md` and `PROCESS.md` are the pilot problem set of Nov 8, which predates the released training data. |
| `paper/` | The writeup, figures, `analysis/jaxbench.py` (every JAXBench table), `data/runs/` (raw rows of every arm) and the notebook. |
| `benchmarks/jaxbench/` | JAXBench's reference files, unmodified (Apache 2.0). They are the exam and the input to the contamination screen, and are never training data. |
| `pallas_guides/` | The Pallas documentation that the docs arms get in front of the task (`teacher_docs.md`), built by `build.py` from the pinned upstream JAX files in `upstream/`. |
| `curation/` | A CPU pre-filter that rejects kernels that cannot lower for a TPU. Its `validate` step reads files that are not in this repository. |

The pipeline scripts under `kimi/`, `dataset/scripts/` and `control/scoring/` are kept as they ran. Several contain absolute paths
and Modal workspace names from the machines they ran on, so they record what was done and need editing before they run elsewhere.
The analysis script, the notebook and the tests run from a clean checkout.

## Provenance

- Oracle verdicts: every per-job verdict file is on branch `hruday/archive-local` under `archive/oracle-results/`. They were
  consolidated from the per-job `oracle-results/*` branches, and `MANIFEST.json` there lists each job branch's head commit.
- Run rows: every run row records its oracle hash, seeds, speed bar, chip, JAX version and the served model's digest.
- Development history: kept on `hruday/archive-local` (scripts and data that lived only on a laptop) and as `archive/*` tags. A
  deleted working branch `<name>` is kept as the tag `archive/<name>`.

## License

- Code (Python, shell scripts and notebooks): [MIT](LICENSE).
- Data, writeup and figures (the dataset releases, `paper/WRITEUP.md`, `paper/figures/`, the run rows and other data files):
  [CC BY 4.0](LICENSE-CC-BY-4.0).
- Model adapter (release `model-final`): CC BY 4.0. It is a derivative of Gemma 4 E4B, so its use is also subject to the base
  model's license and terms.

Third-party material keeps its own license:

- the JAXBench tasks in `benchmarks/`;
- the Pallas documentation excerpts in `pallas_guides/upstream/` (see the `LICENSE` there);
- KernelBench (MIT) and the permissively licensed KernelBook subset.

Kimi K3.1's outputs are subject to its model license.
