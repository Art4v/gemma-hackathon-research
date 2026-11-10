# v4 runs (Hruday's agent, Nov 10 2026)

> **CORRECTION (Nov 10, ~16:00 UTC): the JAXBench directories in the first push were BASE-model runs, not v4.**
> `control/bench.py --backend local` passed `chat_fn = chat`, whose `model` defaults to the base `gemma4:e4b-it-qat`, so `--model` only named the
> run directory. They are kept, renamed, as `base_local__mislabelled_v4qat-e2/` and `base_local__mislabelled_v4final-e2/` (base E4B, Mac Ollama,
> greedy; turn 0 = 0/64 step 1), with their CPU checks in `cpu_checks/base_local_mislabelled/`. The one-line fix to `control/bench.py` is in this
> commit. The v4 val results (`v4_val/`) are unaffected: their scripts pass the model explicitly. The real v4 JAXBench run follows.


**Models.** Two LoRA adapters, both trained quantization-aware: the frozen base is the exact Q4_0 weights Ollama serves (342 Q4_0 + 2 Q6_K + 1 F16 tensors of the official `gemma4:e4b-it-qat` GGUF, sha256 `e8b6a059...`, dequantized bit-for-bit; `v4_training/exact_q4.py`). Recipe otherwise v3's: r64/alpha128 attn+MLP, 2 epochs, lr 1e-4, accum 4 (`v4_training/train_lora_exact_q4.py`). Served on Hruday's Mac Ollama 0.40.14 as the official GGUF with the 258 LoRA tensors stored as dequant(Q4_0)+delta at q8_0 (`scratchpad/fold/fold_local.py`).
- `v4qat` (rehearsal): `aarav/dataset-v4` @ `48cc41b`, 2,806 train rows; val loss 1.08 -> 0.2671. Epoch 2 = `gemma4-e4b-lora:v4qat-e2`.
- `v4final` (headline): freeze build `482b8c4` (= `c2b6499` data), 2,983 train rows; val loss 1.08 -> 0.2601. Epoch 2 = `gemma4-e4b-lora:v4final-e2`.

**JAXBench, first push (INVALID as v4; base-model runs, see the correction above):**
- `gemma4-e4b-lora_v4qat-e2/`: turn 0 complete (0/64 pass step 1); turn 1 partial (stopped when the headline took over).
- `gemma4-e4b-lora_v4final-e2/`: turn 0 0/64, turn 1 0/64 step 1; turn 2 in progress at push time (updated when done).
- `cpu_checks/lora-*-t<N>_static.jsonl`: oracle step 1 on CPU, per attempt (no attempt reached the TPU).

**Val (dataset-v1's 23 val problems):** `v4_val/local_val_*.scored.jsonl` (greedy, CPU step 1) and `v4_val/sampled_*.jsonl` (3 samples per problem at temperature 1.0 / top_p 0.95 / top_k 64, each through the three CPU gates: oracle step 1, v6e lowering, TPU-interpret numerics).

**Training:** `v4_training/*_metrics.json` (full loss/val history).

## The real v4 JAXBench run (Modal, Nov 10, from 17:55 UTC)

**Directory:** `gemma4-e4b-lora_v4final-e2__modal/` (bare, greedy, 3 turns). It is generated on the same backend as base bare/docs and v1:
Modal A10G, Ollama 0.40.14, `control/modal_gemma.py`. The 2 attempts from the paused Mac run are not used.

**Served model, verified before any spend** (`control/modal_v4_create.py`):
- Every manifest layer and the config blob equal the Mac's: model `342ca428`, projector `c6398448`, template `b507b9c2`, params
  `56380ca2`, config `0bb8bebca73f`.
- The Modal digest `0f764d3f6798` differs from the Mac's `2a99a2f97822` only because Ollama 0.40 writes each layer's source path into
  the manifest.
- Every row carries `0f764d3f6798`. The runner stops if any row doesn't.

**62p_Causal_Conv1d_GDN t0.** v4 loops at temperature 0 and Ollama aborts the reply ("token repeat limit reached"), which unstreamed is a
bare HTTP 500. Serving now streams, which gives token-identical output (`control/modal_determinism.py`), and the reply is recorded up to
the abort with `finish: repeat_limit`. It is judged as written: no closed code block, so the oracle's own verdict. Its turn 0 is
judged with turn 1's batch, and one catch-up pass gives it its 3 turns.

**Turn 0** (this push: 64 rows, 63 verdicts, CPU step-1 files `cpu_checks/lora-v4final-e2-modal-t0_*`):
- CPU step 1: 24 of 63 passed (18 k, 6 p). Base bare and v1 had 0 of 64.
- TPU v6e-1, oracle `c9a07d9298a1`: **1 pass**, `12p_RMSNorm`, at 1.40x XLA. The kernel is a single `pallas_call` with 3 reshapes
  outside, correct on 3 seeds, with error equal to XLA default's.
- 2 `correct_slow`: `8p_GEMM` (0.20x) and `48k` (0.16x).
- 4 `wrong`, 16 `vmem` (whole arrays or oversized blocks in VMEM), 1 `compile` (57p).
