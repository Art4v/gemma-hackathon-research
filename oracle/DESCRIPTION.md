# oracle

The oracle runs a solution on a real TPU and says whether it is a Pallas kernel, whether it compiles and runs, whether its output
is correct, and how fast it is compared with XLA. That is the whole definition of the oracle. Dataset curation (the CPU
pre-filter, teacher evaluation of kernels, filtering the dataset) is separate and comes later: see `curation/`.

What is here:

| file | role |
|---|---|
| `tpu_oracle.py` | the verdict on one solution (the four steps below) |
| `tpu_check.py`, `outside_ops.py`, `structure.py`, `keys.py` | what `tpu_oracle.py` imports; with it, the five files whose hash is `oracle_sha` |
| `make_pack.py` | builds a self-contained job directory (oracle files, `pack.json`, `run.sh`, `MANIFEST.json`) |
| `collect.py` | checks a job's results against its manifest and writes `summary.json` |
| `jobq.py`, `estimate.py` | the request queue and a rough chip-time estimate per job |
| `QUEUE.md` | how jobs are run, and the VM security every TPU run needs |
| `manifest.json` | JAXBench's tasks; only `make_pack.py`'s `human_target` check reads it |
| `sandbox/` | the sandbox the `outside_ops` integration test runs in |
| `tests/` | the oracle's tests |

## Current job (Nov 8)

Arav is curating a dataset of tasks and solutions to replace what we have now. The oracle's one job for now is to run every
solution in that dataset on a TPU v6e and give each one a verdict. The JAXBench split (final exam, practice exam, practice
problems) is no longer used and is not part of this job.

### What each solution needs

One line per solution in an items file (JSONL):

```json
{"id": "gemm-001", "task": "gemm", "ref_path": "refs/gemm.py", "code_path": "solutions/gemm-001.py"}
```

- The **reference** (`ref_path` or inline `ref_code`) is plain JAX with `create_inputs()` and `workload(*inputs)`, as in
  JAXBench. `create_inputs` must give the same inputs every time it is called (a seeded RNG), and 3x its input bytes must fit
  in 28 GB.
- The **solution** (`code_path` or inline `code`) defines `workload` with the same arguments, and does its computation inside
  `pl.pallas_call`.
- Ids are unique. All solutions of one task share one reference.

### What the oracle does to each solution

The first failure is the verdict.

1. **Is it a kernel?** Traced without running: there is a `pallas_call`, the output depends on it, and no real computation
   happens outside it. Otherwise `not_a_kernel`.
2. **Does it compile and run?** On the TPU, at the task's full input size. Otherwise `compile`, `vmem`, `mosaic`, `timeout` or
   `runtime`, with the chip's error text in `feedback`. `harness_oom` means the harness failed, not the solution.
3. **Is it correct?** On 3 input seeds, against a high-precision answer (float32 at matmul precision "highest"): within the loose
   tolerance (atol = rtol = 1e-2), or with an error at most 2x XLA's own error. Wrong on any seed: `wrong`.
4. **Is it fast enough?** Timed on seed 0 (3 warm-ups, median of 20 calls) against `jax.jit(reference)` at default precision.
   `speed_vs_xla = t_xla / t_kernel`. `pass` if `speed_vs_xla >= 0.7`, otherwise `correct_slow`.

Each row also records the error per seed (`err_kernel`, `err_xla_default`), the times (`t_kernel_s`, `t_xla_s`, and their p10 and
p90), the `oracle_sha` of the code that judged it, and the chip and JAX version.

### How to run it

```bash
python oracle/make_pack.py ITEMS.jsonl oracle/jobs/JOB --job-id JOB --owner NAME --step STEP   # on the Mac, no JAX needed
bash run.sh --cpu-smoke                                                                         # optional: plumbing check, no TPU
bash run.sh                          # on the v6e VM, as the locked-down user (QUEUE.md, "VM security")
python oracle/collect.py oracle/jobs/JOB                                                        # checks the results, writes summary.json
```

### Not needed for this job

The manifest's groups and `--groups`; `human_target` and `ADAPTERS` (JAXBench's level 2 kernels as speed targets);
`--sensitivity`. `structure.py` and `keys.py` act only on JAXBench task names, so they do nothing for new tasks unless the names
match. They stay in the code and still pass their tests. The CPU pre-filter (`lower_filter.py`) and the JAXBench seed audit
(`int_inputs.py`) moved to `curation/`: neither decides a verdict.

## The speed bar: 0.7x XLA

**0.7 is a policy choice, not a statistical one.** We chose it (Nov 8) to accept correct solutions that are up to about 1.43x
XLA's time (1 / 0.7). The data below shows what statistics can and cannot say about it.

### What the measurements say

From 13 earlier oracle jobs on a real TPU v6e with JAX 0.10.2 (`Gemma-jaxbench-study`, `cloud-gemma/sera-e4b/oracle/results/`):

- **Timing noise is about 3%.** Every job times XLA, so the same reference was timed in 2 to 4 separate sessions. Across 32
  JAXBench tasks (0.3 ms to 96 ms per call), the same reference's XLA time differed by 0.0% to 3.5% between sessions, usually
  by under 1.5%. (One task differed by 114x, but its two jobs used different reference files, so it is not noise.)
- **Google's own Pallas kernels run at 1.6x to 5.0x XLA:** 11p Megablox GMM 1.60x, 6p Paged Attention 2.35x,
  7p Ragged Paged Attention 5.00x (2p GQA ran at 2.45x but was wrong on a seed).
- **Model-written kernels:** of 72 correct ones (`s1-human-and-agreement-v2`), the median was 0.63x XLA; 23 were at
  0.97x or above, 31 at 0.8x or above, and 34 at 0.7x or above.

### What that means for the bar

- With noise of about 3%, a solution between about 0.97x and 1.03x cannot be told apart from XLA. A bar justified by the
  statistics alone would be about **0.97x: "at parity or faster"**.
- A bar of 0.7 is well below that noise floor. A solution at 0.7x is measurably slower than XLA, not slower by chance. The bar
  is a decision about which slower solutions we still accept, and we state it as one.
- So the parity rate stays visible: reports should give the share of solutions at `speed_vs_xla >= 0.97` alongside the pass
  rate at 0.7. Every row records `speed_vs_xla`, so both come from the same results.

### Limits of this evidence

- Most tasks were timed in only 2 to 4 sessions: enough to say "a few percent", not to fix an exact 97.5th percentile.
- It measures XLA's run-to-run noise. Pallas kernels should behave alike, but that has not been measured.
- The shortest task took about 0.3 ms. Much smaller kernels are likely noisier, because the timer includes the time to launch
  work on the chip.
- XLA and the solution are timed at different moments in a run, not alternately. Drift between the two would show in the
  ratio; across sessions it stayed under 3.5%.
- One reference-only job on the v6e (about 30 to 40 minutes of chip time) would tighten these numbers if needed.

`--speed-bar` changes the bar for a job; each row and the job's MANIFEST.json record the bar it was judged with.
