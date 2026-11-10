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
