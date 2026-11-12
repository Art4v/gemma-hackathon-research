# Folding and scoring a LoRA adapter the way v4 and v5 were scored

**1. Fold** the adapter into the official QAT Q4_0 GGUF (LoRA delta added at q8_0):

```
python fold_local.py <base blob> <adapter dir> <out.gguf> --dtype q8_0
```

- **Base blob:** `~/.ollama/models/blobs/sha256-e8b6a059ba86947a44ace84d6e5679795bc41862c25c30513142588f0e9dba1d`. This is the 5.15 GB model layer of `gemma4:e4b-it-qat`, and its sha256 is the file name.
- **Dependencies:** numpy, gguf, transformers.

**2. Register** the GGUF in the `gemma-bench-ollama` Modal volume:
- **Script:** `control/modal_v5_create.py`, with `NAME`, `GGUF` and `GGUF_SHA` set for the new model.
- **Configuration:** Ollama 0.40.14, the same projector and Modelfile lines as on the Mac.
- **Check:** the script verifies the GGUF sha256 and the manifest layers. Record the resulting model digest.

**3. Score** on JAXBench with `jaxbench_modal.sh`:

```
MODEL=gemma4-e4b-lora:<tag> PREFIX=lora-<tag>-modal EXPECT_DIGEST=<digest prefix> MODAL_PROFILE=<profile> BUDGET_CAP=<usd> bash jaxbench_modal.sh
```

- **Prompt:** the bare prompt, greedy decoding, seed `crc32`.
- **Generation:** `control/bench.py --backend modal`, Ollama 0.40.14 on a Modal A10G, `num_ctx` 65536, `max_tokens` 8192, thinking off.
- **Turns:** 3 in total: turn 0 plus 2 repair turns. After every generate step, every row must carry `EXPECT_DIGEST`.
- **Judging:** CPU step 1 (`control/cpu_check.py`), then the TPU oracle `c9a07d9298a1` on a v6e-1 (`crashloop.py`).
- **Ingest:** only each task's final verdict is ingested. Halt-cascade victims are skipped and parts are read in numeric order.
