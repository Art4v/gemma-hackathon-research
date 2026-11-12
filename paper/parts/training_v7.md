# v7 training: facts for the writeup

Written by Aarav's agent from the run's own files. Every number here is from `runs/v7final_r64/metrics.json` (copied to
`gs://gemma-hackathon-adapters-5ec089ff/v7/final/adapter/metrics.json`), the trainer source, or the launch and billing output.

## What was trained

- **Data:** `v7_final.jsonl` @ `hruday/kimi-k31` `2efa6a5`, 2,304 rows, sha256 `10345461a48148d379f99c7d304180eff0d23bf9693568cbef99090db7abcda9`.
  - **Validation:** `v7_final_val.jsonl`, 29 rows (the dev holdout), sha256 `d09e86649892595c375769c7c779deeefa9fcaecf9a140f755b87b0166860fac`.
  - **Counts:** the trainer reported `examples` 2,304 and `val_examples` 29. `near_test_rows` was 0, so nothing was filtered or held out.
- **Base:** Gemma 4 E4B instruction-tuned, trained on the exact Q4_0 weights Ollama serves: the official `gemma4:e4b-it-qat` GGUF,
  dequantised bit for bit, so training and serving see the same base.
- **Adapter:** LoRA rank 64, alpha 128, dropout 0, on the attention and MLP projections of all 258 target layers.
  - **Size:** 139,526,144 trainable parameters.
  - **Merge check:** 2,076 of 2,076 base tensors matched, 258 changed, none missing or mismatched.

## How

- **Trainer:** `control/runs/v4_training/train_lora_exact_q4.py` on `hruday/v4-runs`, the same script as v4final, v5 and v6.
  - **Local copy:** it differs in two lines. One is the Modal app name. The other is `gpu=...split(",")`, so Modal could take the first free GPU from a list during the capacity shortage; this doesn't change training.
- **Command:** `--epochs 2 --lr 0.0001 --accum 4 --max-tokens 24576`, with `--val` given.
- **Loss:** cross-entropy on the final assistant turn only (the kernel), averaged over its tokens. Prompt tokens and earlier failed attempts in repair rows are context, not targets.
- **Long rows:** a row over 24,576 tokens would be dropped whole; none was. The longest was 11,854 tokens.
- **Schedule:** 4,608 example steps = 1,152 optimiser steps (gradient accumulation 4).
  - **Optimiser:** AdamW, weight decay 0.01, gradient clipping at 1.0.
  - **Learning rate:** 10% linear warm-up, then cosine decay to 0. Peak lr is 1e-4.
  - **Order:** examples shuffled with `random.seed(0)`.
- **Hardware:** Modal, one NVIDIA A100-SXM4-80GB (`nvidia-smi` in the container), 8 CPUs, 64 GB RAM.
  - **Time:** 5,350 s wall time (89 min); 5,113 s inside the training loop.
  - **Memory:** peak GPU memory 30.5 GB.
- **Scored adapter:** fixed before training: the end of epoch 2 (`adapter`, sha256 `a1259bb6…2243`). `adapter_e1` (`93bc1af2…af05`) was saved as a fallback and is not scored.

## Loss

Per-example training loss, from `loss_v7.csv` (one row per example step: epoch, task, tokens, loss, lr, seconds):

| Examples | Mean training loss |
|---|---|
| 1–100 | 0.664 |
| 101–500 | 0.207 |
| 501–1,152 | 0.133 |
| 1,153–2,304 (rest of epoch 1) | 0.110 |
| 2,305–3,456 | 0.076 |
| 3,457–4,608 | 0.074 |

- **Epoch means:** 0.157 (epoch 1) and 0.075 (epoch 2); the last 100 examples average 0.069.
- **Dev validation loss:** 1.318 before training and 0.0002 after.
  - **Readings logged during the run:** 0.521 at example 100, 0.018 at 300, 0.005 at 500, then 0.0001–0.004 for the rest.
  - **Caution:** the 29 dev items are generator-template kernels, held out by normalised op chain. Training contains hundreds of rows in the same template, so once the template is learned a dev item's code is almost fully predictable from its reference. This validation loss therefore says nothing about tiling ability; only the TPU results do.

## Cost and what went wrong

- **Spend:** Aarav's Modal went from $59.56 metered (03:22 UTC) to $65.30 (05:47 UTC), so **$5.74** for v7 training. That includes about 17 minutes of an aborted run.
- **Earlier runs:** v5full (2,653 rows, H100) cost about $6.60, and v6 (1,332 rows, H100) about $2.70.
- **GPU capacity:** a first launch at 03:30 UTC on an H100 sat queued for 14 minutes with no container. A relaunch on an A100-80GB also queued; Modal reported it was "waiting to be scheduled". With the GPU list, an A100 came up at 03:53.
- **The lost insurance run:** that 03:53 run trained the earlier 1,717-row file as insurance. It died with "torch can only be initialized once per process" right after the app was redeployed and the final run launched. It produced no adapter and nothing is scored from it.
  - **Lesson:** don't redeploy or relaunch on a Modal app while a call on it is live.
- **The final run:** it got a container at 04:10:49 UTC and ran uninterrupted.
