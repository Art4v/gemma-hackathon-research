# gemma-hackathon-research

Teaching Gemma 4 E4B, a model small enough to run on a laptop, to turn plain-JAX code into Pallas TPU kernels that are correct and
fast on a real TPU v6e. Built for the Kaggle Gemma 4 Developer Agent Paper Track (writeup due Nov 12, 2026).

## The problem

Custom Pallas kernels can beat XLA by large factors, but Pallas is young, thinly documented, and strict about block shapes and
layouts. Small open models fail at it: on JAXBench's 64 tasks, base Gemma 4 E4B solves 0, with or without 25k tokens of Pallas docs
in the prompt. We fine-tune it on kernels that a real TPU has verified.

## Status (Nov 10, 2026)

| | |
|---|---|
| JAXBench baselines, 3 attempts per task | base Gemma 0/64, Gemma + docs 0/64, Gemini Flash + docs 8/64 |
| First LoRA (`v1_r64`, 238 rows) | 0/64; it learned the format but not the API. The 4-bit fold also erased most of the adapter (fixed with an F16 fold) |
| v3 adapters (1,597 and 2,877 rows) | trained; validation pending |
| Dataset v4 | release candidate out (`dataset-v4-rc1`); freeze Nov 11 06:00 UTC, then v4 training and the JAXBench run |

The pre-registered evaluation protocol is `control/PROTOCOL.md`.

## Datasets

| Tag | What it is |
|---|---|
| [`dataset-v4-rc1`](https://github.com/Art4v/gemma-hackathon-research/releases/tag/dataset-v4-rc1) (pre-release) | 2,717 train / 26 val rows over 1,398 tasks; 2,525 targets TPU-verified. Zip and tar.gz with a datasheet, checksums, licenses and per-task attribution |
| `dataset-v1` | 238 train rows, the data behind the first adapter |
| `dataset-v0`, `dataset-pilot-r1-teacher` | early pilots |

Each training row is a chat built by the evaluation's own prompt function, so it is byte-identical to what the model sees at test
time. A row is either one problem and a passing kernel, or a repair chain: a failed kernel, the oracle's feedback verbatim, and the
fix.

`notebooks/dataset_v4_companion.ipynb` downloads the release, checks every file against its checksum, reproduces the counts and
the oracle-result breakdowns, and runs the oracle's kernel check on CPU. It needs no TPU.

## The oracle

`oracle/` judges a solution on a TPU v6e at the task's full input size. The first failing step is the verdict:
1. **Is it a kernel?** There must be a `pallas_call` the output depends on, with only layout, dtype and small integer work outside it;
   otherwise `not_a_kernel`. This step is static and runs on CPU.
2. **Does it compile and run?** Otherwise `compile`, `vmem`, `mosaic`, `timeout` or `runtime`, with the chip's error as feedback.
3. **Is it correct?** On 3 seeds against a float32 answer key; otherwise `wrong`.
4. **Is it fast enough?** At least 0.7x XLA (median of 20 timed calls); otherwise `correct_slow`.

Every result row records the hash of the three oracle files (`c9a07d9298a1`), the seeds, the speed bar, the chip and the JAX version
(0.10.2). Verdicts from v6e-1 and v6e-8 hosts agree: on 22 kernels judged both ways the verdicts matched and the median speed ratio
between hosts was 0.997.

## How the data is made

1. **Problems.** KernelBench ports, operators written for this project (attention variants, MoE routing, quantized matmuls, KV-cache
   updates, SSM scans), references for public Pallas kernels, and PyTorch modules from GPU MODE's KernelBook ported to JAX. Each
   problem passes CPU readiness checks at full size.
2. **Contamination screen** against all 73 JAXBench files: KernelBench-stem match, jaxpr op-chain graph signatures, near-duplicate
   text, and an algorithm review that excludes problems sharing a JAXBench task's core algorithm.
3. **Kernels.** A constructor model writes each kernel from the problem and the Pallas docs in `pallas_guides/`, and lowers it for
   TPU on CPU first. A failing kernel gets up to two repairs, each prompted with the oracle's verbatim feedback.
4. **Judging** on TPU v6e, then collection into one canonical job per batch and round.

The repair rounds taught us one thing worth knowing: most correct-but-slow kernels were not slow inside the kernel. A reshape outside
`pallas_call`, such as `(B,C,H,W)` to `(B,C,H*W)`, is not a free relabelling in the TPU's tiled layout, so XLA inserts a full copy of
the input or output around the kernel. Rewriting kernels to keep native layouts recovered 26 of 51 that had already failed twice.

## Repository layout

| Path | What is there |
|---|---|
| `oracle/` | the TPU oracle (`tpu_oracle.py`, `outside_ops.py`, `tpu_check.py`), job packing (`make_pack.py`), VM setup (`vm.sh`) |
| `control/` | the evaluation harness (`bench.py`), the protocol (`PROTOCOL.md`) and verified JAXBench references |
| `benchmarks/jaxbench/` | the JAXBench tasks used for evaluation (its own license) |
| `curation/` | the CPU cross-lowering filter that rejects kernels before they reach a TPU |
| `dataset/` | problems, candidates, prescreen results, verdicts, trajectories and the build scripts |
| `pallas_guides/` | the Pallas documentation given to the constructor and to the docs baseline |
| `training/`, `results/` | training notes and result records |
| `notebooks/` | the dataset and results companion notebook |

## Branches and tags

- `main`: the stable tree.
- `dataset-gen`: the dataset-v1 lineage that the data branches build on.
- Data branches by owner, holding the canonical collections v4 is assembled from: `karthik/kbk` (KernelBook and the production
  set), `karthik/kbkc` (KernelBook shard C), `karthik/m1` (structural mutations), and `aarav/dataset-v4` (the v4 assembly).
  The `hruday/*` and `aarav/*` branches carry each person's kernels and repairs.
- `oracle-results/<job>`: the raw oracle output for each judged job. These will be folded into one `oracle-results` branch at the
  data freeze.
- `archive/<branch>` tags: retired branches, kept so no commit is lost.

## Running the oracle

On a GCP project with v6e quota (on-demand only; delete the VM when the queue is empty):

```
export PROJECT=<project> ZONE=<zone> NAME=oracle-v6e
bash oracle/vm.sh create && bash oracle/vm.sh bootstrap && bash oracle/vm.sh check
python oracle/make_pack.py items.jsonl oracle/jobs/<job> --job-id <job> --owner <you> --step <job>
bash oracle/vm.sh push oracle/jobs/<job> && bash oracle/vm.sh run <job>
bash oracle/vm.sh status <job>; bash oracle/vm.sh fetch <job>
bash oracle/vm.sh delete
```

`oracle/tools/` on `karthik/kbk` adds a capacity hunter (`grab_v6e.sh`), a crash-splitting runner (`crashloop.py`) and a job loop
(`ops_loop.sh`); `vm.sh` there also takes `ACCEL=v6e-8`, with `make_pack.py --shards 8` putting one shard on each chip.

## Licenses

KernelBook-derived problems are used under the June 9 Researcher Reciprocity License; its Attachment A applies to models trained on
them. KernelBench-derived problems are MIT. JAXBench and the upstream Pallas docs keep their own licenses (`benchmarks/jaxbench/LICENSE`,
`pallas_guides/upstream/LICENSE`). The dataset release carries the full texts and a per-task attribution file. The team has not yet
chosen a license for its own code and authored data.

## Team

[Art4v](https://github.com/Art4v), [hnaracodes](https://github.com/hnaracodes) and [LordKarV](https://github.com/LordKarV). Much of
the work was done by each person's Claude Code agent, coordinating on issues #3 (TPU job claims) and #5.
