# Project status (Nov 9, ~02:10 EST): `dataset-v1` FROZEN

**In one paragraph:** We're fine-tuning a small Gemma 4 model (E4B) to turn plain JAX code into fast Pallas TPU kernels, then showing on JAXBench that it beats the untrained model. The training **dataset is done and frozen as `dataset-v1`** (tag on `dataset-gen`, commit a0e6770): **236 solved problems, 264 training examples**, all checks passing. Next: **train the LoRA** on your new Modal account. Deadline: the Kaggle writeup is due **Nov 12 23:59 UTC**.

---

## 1. The pipeline, step by step

```
problems (plain JAX)  ->  teacher writes a Pallas kernel  ->  CPU prescreen  ->  TPU oracle judges it
        ^                                                                         |
        |                     fail? repair with the oracle's exact feedback  <----+   (max 3 attempts)
        |
   pass (correct on 3 seeds AND >= 0.7x XLA speed)  ->  training example  ->  dataset-v1  ->  LoRA on Gemma 4 E4B
```

1. **Problems.** Each is a plain-JAX reference (`workload(*inputs)`), from **KernelBench ports**, **authored** modern-model operators (attention variants, MoE, quantization, KV cache, SSM), and a few **scraped** kernels. They are filtered for contamination against the 65 JAXBench tasks (op-graph similarity), and **CPU-checked**: deterministic, inputs vary by seed, fits TPU memory, runs.
2. **The teacher writes a kernel**, given the oracle's rules plus the Pallas TPU docs.
3. **Prescreen:** the oracle's own static check, run on CPU. Is it a real Pallas kernel, and does it trace?
4. **The TPU oracle** (TPU v6e, oracle code `c9a07d9298a1`, frozen) judges it: compile → correct on 3 seeds vs a float32 answer key → speed vs XLA. **Pass = correct AND ≥ 0.7x XLA.**
5. **Repairs:** if it fails, the teacher sees the oracle's **verbatim feedback** and tries again. **At most 3 attempts** (r1, r2, r3); after that the problem is "unsolved".
6. **Assemble:** every solved problem becomes a training example. A problem solved on a repair also yields a **repair chain** (kernel → feedback → fix), which teaches the model to use feedback the way the eval does (3 turns).

Batches (`pilot`, `b1`, `b2`, `k1`) are just groups of problems sent to the TPU as one job (`dataset-<batch>-r<round>`).

---

## 2. What happened tonight, in order

| When (EST) | What |
|---|---|
| Nov 8 evening | Aarav's driver had generated the pilot + b1 kernels with **Claude** (`claude -p`). |
| ~17:30 | **Licensing question:** Anthropic's Usage Policy prohibits *"model distillation"* without authorization. The team switched the teacher to **Gemini**. |
| 17:30–19:00 | Built a Gemini pipeline (`dataset-gemini` branch): argon wrote all problem references, Flash-Lite wrote the kernels (cheapest), g1–g3 went to the TPU. |
| ~19:00 | **Gemini credit ran out** (HTTP 402, ~$20 spent). Flash-Lite's quality was poor (16% pass, mostly compile errors). |
| ~20:00 | **Karthik reversed the decision:** back to the **Claude dataset** (`dataset-gen`), on the basis that Aarav's generation ran on the paid API. The Gemini work is retired, kept on its branch as a record only. |
| 20:38–21:40 | Resumed Aarav's driver here. `claude -p` **hung**: a Playwright MCP server from your Claude Code config started with every call. **Fixed** (`--strict-mcp-config`, process-group kill on timeout). Generation was still slow on the subscription. |
| ~21:43 | **Switched to subagents** for kernel generation: 6 in parallel, each checking its kernels with the oracle's static check. b2's 89 kernels done in ~20 min. |
| 22:03–22:15 | b1-r2 (11) and b1-r3 (2) queued; **Aarav's agent ran them** on his GCP TPU. **b1 final: 117/119.** |
| 22:43 | **Karthik's agent added k1:** 17 problems that had failed *our* CPU check (Mac memory limit), with kernels written on his side and judged on his TPU: 13 pass. Folded into our dataset. |
| 22:43–22:50 | A stale second driver was running (my mistake); found and killed it. State repaired. |
| 23:50–00:24 | b2-r1 on Aarav's TPU (a spot VM was preempted once, rerun on-demand): **64 pass**, 21 correct-but-slow, 2 compile errors. 2 items crash the TPU compiler and are excluded. |
| 00:26–00:33 | b2-r2: 22 repairs by subagents → **11 more pass**. k1-r2 → 1 more pass. |
| 01:05–01:39 | Round 3 (last try): k1-r3 (3) → all correct-but-slow, so unsolved. b2-r3 (11) queued. |
| 01:49 | b2-r3 back: 3 more pass (**b2 final 78/87**). The driver assembled + verified the final set. |
| ~02:05 | Contamination + near-dup filters clean → **tagged `dataset-v1`**, announced on #5. Asked Aarav's agent to delete its TPU. |

---

## 3. The dataset right now

Branch **`dataset-gen`** (Aarav's Claude dataset), driver running on this Mac (`~/Projects/gemma-dataset-claude`).

| Part | Problems | Solved | Unsolved | Status |
|---|---|---|---|---|
| pilot | 30 (+4 excluded) | 29 | 1 (+2 left open, out of rounds) | done |
| b1 | 119 (+1 excluded) | **117** | 2 | done |
| k1 | 17 | **14** | 3 | done |
| b2 | 87 (+2 excluded) | **78** | 8 | done |
| **Total** | | **236** | 14 | **frozen** |

**`dataset-v1` (final export, `dataset/final/`):**
- **264 examples** (238 train / 26 val): 236 single-turn + **28 repair chains**
- Sources: 123 authored · 138 KernelBench · 3 scraped
- First-try pass rate 82% (pass@3: 93%); **83% of solved kernels at or above XLA speed** (≥ 0.97x)
- **`verify_dataset`: OK** · contamination filter: 0 new exclusions, 0 review · near-dup filter: 0 (max Jaccard 0.49)
- The split keeps problem families together, so a problem and its bf16 variant can't straddle train/val.

**Excluded (status `excluded`, never in the export):**
- 3 harness crashes: the oracle itself dies in XLA's TPU compiler
- KLDivLoss: its inputs don't vary across the 3 seeds
- 3 JAXBench lookalikes (op-graph similarity 0.93–0.95): `ConvTranspose3d_BatchNorm_Subtract`, `Conv2d_Mish_Mish` and its bf16 variant. **Your decision.**
- `assemble.py` was patched to skip excluded problems; before that, it would have emitted them anyway.

**Skipped:** the bf16 variants (~120 problems). Optional, too slow on time; never generated.

---

## 4. Checks run on the training rows (on the pre-round-3 export; rerun on `dataset-v1` before training)

Rows built with `training/build_rows.py`, which builds every prompt with `control/bench.py`'s own `conversation()`, the same function the eval uses.

| Check | Result |
|---|---|
| Template parity: each training prompt vs `bench.conversation` rebuilt from the raw turns | ✅ **258/258 byte-identical** |
| Repair feedback rendered verbatim | ✅ all 25 chains (max 2 repairs) |
| Targets = the exact TPU-passing kernel | ✅ all 258 |
| Tokens (Gemma 4 tokenizer + HF chat template), prompt + target | ✅ median 2,140 · p99 5,795 · max 6,957 (training cap 24,576: **0 dropped**) |
| Tokens, kernel reply only | ✅ median 1,221 · p99 3,475 · max 5,921 (eval generation cap 8,192: **all fit**) |
| Ollama's template vs HF's template | ⏳ not yet, needs the Modal preflight (it compares against the server's tokenization) |

---

## 5. People, machines, accounts

| Who / what | Role tonight |
|---|---|
| **Your Mac** | Runs the dataset driver (`dataset-gen`): collects results, builds jobs. Kernels are written by Claude subagents in this session. |
| **Aarav's agent** (posts as Art4v) | Runs TPU jobs on a v6e on **Aarav's GCP credit** ($300). Asked to **delete the VM** right after b2-r3. Estimated spend ~$8–13 (real figure requested). |
| **Karthik's agent** (posts as LordKarV) | TPU backup: on-demand v6e on Karthik's GCP for the LoRA eval. Writes the paper draft (`karthik/writeup`), stats script, `PROTOCOL.md`. Karthik is offline for a few hours. |
| **Modal `codecraftacademy981`** | Your new account for LoRA training + eval generation. The Ollama model is pulled, and its **digest matches the baseline run** (`ee66563…`, Ollama 0.40.14). |
| Colab TPU | Never got a v6e ("Service Unavailable"). Stopped, nothing billed (92.64 units left). |
| Gemini key | Out of credit. Gemini work retired. |

Coordination happens on GitHub **issue #3** (TPU jobs), **#5** (plan to submission), **#2** (benchmark protocol).

---

## 6. Decisions made (and recorded on GitHub)

- **Teacher = Claude** (Karthik's call), with per-row provenance (`claude_cli` from Aarav's run, `claude_code_subagent` for tonight's subagent kernels).
- **A2 = no:** eval stays at **1 greedy chain × 3 turns** for every Gemma arm, matching the existing bare/docs baselines, so no reruns. Limitations to state: the Flash arm samples (Gemini loops at temperature 0), and greedy decoding is prone to repetition loops.
- **Exclusions** (above): KLDivLoss + the 3 JAXBench lookalikes.
- **Control runs pushed:** branch `hruday/control-runs` (base model: bare 0/64, docs 0/64 within 3 turns).
- **Variants skipped.**

---

## 7. What's next

**Data: ✅ done.** `dataset-v1` is tagged and announced. Waiting for Aarav's agent to confirm the TPU is deleted and post its spend.

**Training (your H4, next):**
4. Preflight on Modal `codecraftacademy981` (Ollama template parity, one training step).
5. Train the LoRA on `dataset-v1`: r64 / alpha 128, attention + MLP, 3 epochs, lr 1e-4 (the pilot25 setup). The 25-row pilot took 108 s; ~234 rows is minutes.
6. Fold the LoRA into the Ollama GGUF (`training/to_ollama.py`) → serve → **eval on JAXBench** (`control/bench.py`, 1 greedy chain, 3 turns): `lora` and `bare_stack`. TPU judging via Karthik's agent (or Aarav's).
7. Fill in `PROTOCOL.md`'s 3 TODOs (stack, hyperparameters, headline adapter) **before** the first LoRA number.

**Writeup and publishing (parallel):**
- Section 3.5 (training details) in `karthik/writeup`.
- Provenance: the dataset is Claude-generated. State it plainly, since this is where Anthropic's Usage Policy applies to a public release.
- Update `dataset/README.md` / `PROCESS.md` (k1, exclusions, subagent rounds, the mix of constructors).
- Check the jagged-softmax "776x" outliers (probably slow references); report speed by median or separately.
- Optional: reverify a 10% random sample on the TPU (Karthik's agent offered).

---

## 8. Glossary

- **Oracle:** the frozen judge (`tpu_oracle.py`). Kernel check → compile → correct on 3 seeds → speed vs XLA.
- **pass / correct_slow / compile / mosaic:** passed both bars / correct but under 0.7x / didn't compile / Pallas TPU compiler error.
- **Round (r1–r3):** attempt number for a problem. r2/r3 are repairs with oracle feedback.
- **Subagent:** a parallel Claude worker spawned in this session to write kernels.
- **Driver:** `dataset/scripts/overnight.py`: collects TPU results, builds and queues jobs, keeps state in `overnight_state.json`.
- **bare / docs / lora / bare_stack:** eval arms. Base model without docs / with docs / our fine-tune / base model on the LoRA's exact serving stack.
