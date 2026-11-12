# Training

`train_lora_exact_q4.py` is the LoRA trainer for Gemma 4 E4B, kept byte-identical to the copy that ran on Modal (sha256
`c766a0e97cb19266…`). It trains against the exact weights Ollama serves: the Q4_0 tensors of the official `gemma4:e4b-it-qat` GGUF
(sha256 `e8b6a059…`), dequantized, instead of the bf16 checkpoint (`EXACT_Q4 = True`). A LoRA trained on bf16 weights and rounded back
into Q4_0 is mostly erased; see the writeup, Section 3.5.

- **Adapter:** rank 64, alpha 128, on the attention and MLP projections.
- **Loss:** the final assistant turn only. Rows over 24,576 tokens are dropped whole.
- **Final adapter** (run name `v7final_r64`):
  - data: `train_as_trained.jsonl` (2,304 rows) and `val.jsonl` (29) from the release `dataset-final`;
  - `--epochs 2 --lr 1e-4 --accum 4`, on one A100-80GB, 89 minutes;
  - adapter `adapter_model.safetensors` sha256 `a1259bb61e5f6551…`.

```
export MODAL_PROFILE=<workspace>
modal run training/train_lora_exact_q4.py --workspace <workspace> --step preflight --data train_as_trained.jsonl
modal deploy training/train_lora_exact_q4.py
modal run training/train_lora_exact_q4.py --workspace <workspace> --step launch --data train_as_trained.jsonl --val val.jsonl \
    --name <run> --epochs 2
```

The file's docstring keeps the history of where it came from, including the example paths of the runs it was first written for.
Serving: `control/scoring/fold_local.py` folds the adapter into the GGUF (the 258 touched tensors at Q8_0, every other tensor
byte-identical), and `control/scoring/` runs JAXBench on it.
