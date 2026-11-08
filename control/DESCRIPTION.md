# control

The controls of the benchmark: base Gemma, with no fine-tuning, writing Pallas TPU kernels for JAXBench, judged by the TPU oracle
(`oracle/`). The LoRA arm (Aarav's dataset) will run through the same harness with the same settings, so the three arms differ in one
thing each.

| arm | model | prompt |
|---|---|---|
| `bare` | `gemma4:e4b-it-qat` (Ollama) | the task's plain-JAX reference only |
| `docs` | the same | `pallas_guides/teacher_docs.md` (about 25k tokens), then the same prompt |
| `lora` (later) | the same base + the LoRA | the same as `bare` |

## Tasks

64 of JAXBench's 65 tasks:
- **50 level 1 tasks**: their own plain-JAX `benchmarks/jaxbench/level1/<task>/baseline.py`, unchanged.
- **14 level 2-only tasks (51p-54p, 56p-65p)**: JAXBench has only Google's Pallas kernel for these, so the prompt needs a plain-JAX
  reference: `control/refs/<task>.py`. Each one's `create_inputs` is copied verbatim from Google's file (the same inputs); its
  `workload` is plain JAX, written by Gemini in planningGemma (step 5, `jaxbench/runs/step5/refs.jsonl`, Nov 7). They are checked on
  the TPU against Google's kernel before any result counts (`verify_refs.py`, below).
- **55p is left out**: its output is a top-k index *set* in unspecified order (JAXBench sorts both outputs before comparing), but the
  oracle compares outputs element by element, so a correct kernel could be marked wrong.

## Settings (every arm)

- One sample per task, Ollama `/api/chat`, thinking off, temperature 0, seed `crc32("<task>/0/<turn>")` (the same for every arm),
  context 65,536 tokens, at most 8,192 output tokens (a reply is 1-2k tokens; at temperature 0 a reply can repeat itself until the
  cap, which then counts as a reply with no code).
- Prompts: planningGemma's `rounds.conversation` and `rollouts.repair_turn`, word for word (`bench.py`: `SYSTEM`, `BEFORE_REF`,
  `AFTER_REF`). The LoRA's training data should use the same format.
- Turns: the first attempt (turn 0) and up to 2 retries. A retry sees every earlier attempt with the oracle's `feedback` line on it,
  then "Reply with the corrected Python code block." A task stops at `pass`, at `too_large_for_chip` or `harness_oom` (not the
  kernel's fault), or after turn 2. The code goes to the oracle as extracted, so a reply with no code block gets the oracle's own
  verdict on it ("no attribute 'workload'"), and that is the feedback its retry sees.
- Judged by the oracle at its defaults: 3 seeds, speed bar 0.7.

Every row (`control/runs/<model>/<arm>.jsonl`) records the item id (`<arm>/<task>/t<turn>`), the reference's sha256, the prompt
template's sha256, seed, sampling settings, token counts, finish reason, time, the full reply and the extracted code.

## Running it

On the Mac (Ollama running):
```bash
python3 control/bench.py generate --arm bare      # every task's next attempt; resumable
python3 control/bench.py generate --arm docs
python3 control/bench.py items --out control/runs/items_t0.jsonl      # both arms' attempts waiting for the oracle
python3 oracle/make_pack.py control/runs/items_t0.jsonl oracle/jobs/control-t0 --job-id control-t0 --owner hruday --step t0
python3 control/verify_refs.py build oracle/jobs/verify-refs-1 --job-id verify-refs-1 --owner hruday    # once
```
On a Colab TPU v6e session (compute units from `colab pay`), with `oracle/colab.sh`:
```bash
bash oracle/colab.sh start && bash oracle/colab.sh setup
bash oracle/colab.sh push oracle/jobs/verify-refs-1 oracle/jobs/control-t0
bash oracle/colab.sh run verify-refs-1      # then status until it's done; one job at a time on the one chip
bash oracle/colab.sh run control-t0
bash oracle/colab.sh fetch verify-refs-1 && bash oracle/colab.sh fetch control-t0
bash oracle/colab.sh stop                   # always: an idle session burns compute units
```
Or on a Cloud TPU VM, as the locked-down user (`oracle/QUEUE.md`, "VM security"), with `oracle/vm.sh` from a machine with gcloud:
```bash
export PROJECT=<project id> ZONE=<zone with v6e> SPOT=1
bash oracle/vm.sh create && bash oracle/vm.sh bootstrap && bash oracle/vm.sh check
bash oracle/vm.sh push oracle/jobs/verify-refs-1 oracle/jobs/control-t0
bash oracle/vm.sh run verify-refs-1      # the reference check first; status shows progress, fetch brings the results back
bash oracle/vm.sh run control-t0         # after verify-refs-1 finishes (one chip, one job at a time)
bash oracle/vm.sh fetch verify-refs-1 && bash oracle/vm.sh fetch control-t0
bash oracle/vm.sh delete                 # when the round is done: the VM bills idle
```
Back on the Mac:
```bash
python3 control/verify_refs.py report oracle/jobs/verify-refs-1/results/verify-refs-1.jsonl
python3 oracle/collect.py oracle/jobs/control-t0
python3 control/bench.py ingest oracle/jobs/control-t0/results/control-t0.jsonl
python3 control/bench.py generate --arm bare && python3 control/bench.py generate --arm docs    # turn 1, then items, a job, ingest; turn 2 the same
python3 control/bench.py summary
```
A level 2-only task whose reference fails the check is left out of the results until its reference is fixed.

## Results

`bench.py summary` per arm: tasks judged, `pass_at_1` (turn 0), `pass_within_3`, `correct_within_3` (pass or correct_slow),
`parity_within_3` (correct and at least 0.97x XLA, see `oracle/DESCRIPTION.md`), the median speed of the correct ones, and the verdict
counts at turn 0 and at each task's end.
