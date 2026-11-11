# Kimi K3.1 kernel dataset (first attempts), Nov 11 2026

Every problem in the pipeline's pool, one first attempt each by **Moonshot Kimi K3.1** (Pro Highspeed). The model was called through Amazon
Bedrock (`global.` / `us.moonshotai.kimi-k3.1`, Converse API, `reasoning_effort` high, max 48k output tokens) on Hruday's AWS account.
Each row keeps Kimi's **reasoning text, plan and code**. Every kernel was then judged by the oracle: first oracle step 1 on CPU, then
the step-1 survivors on a real **TPU v6e** (oracle sha `c9a07d9298a1`).

## Problems (`data/full_problems.jsonl`, 1,672)
- **Pool:** every `status: ok` problem from the five problem files: dataset-gen (Claude era), kbk-r1, kbkc-r1, m1-r1 and kbk-c.
  They are deduplicated by problem_id, then by reference sha (`scripts/build_full.py`).
- **Excluded:**
  - every variant of the 23 dataset-v1 val tasks (115 problems)
  - the 88 `decision: exclude` rows of the algorithm-level contamination review (`algo_review_kbk.jsonl`)
  - anything named like a JAXBench task (none matched)
- **By source:** kernelbook 1,214, kernelbench 281, authored 160, github 15, scraped 2.
- **Prompt:** identical to the Claude constructor's round 1. The system prompt is prompts-1 plus `pallas_guides/teacher_docs.md`
  (`scripts/system_prompts-1_with_docs.txt`, sha `dc7c4acc…`), with one appended rule asking for a single JSON `{plan, code}`. The
  user turn is `render_chat.render_problem(ref_code)` (render-1).

## Outputs (`data/out/`)
- **`k31_*.jsonl`:** one row per attempt, with these fields:
  - `problem_id`, `task_name`, `source`, `family_id`
  - `ok`, `reasoning` (Kimi's reasoning text, median ~26k chars), `plan`, `code`
  - `usage` (tokens, cost at Bedrock list price), `profile`, `effort`
  - 1,672/1,672 problems have a parsed kernel. 7 malformed first answers were re-run (`k31_full_us_s0.jsonl`).
- **`cpu_scored.jsonl`:**
  - oracle step 1 (`tpu_oracle.py --static-only`, CPU, JAX 0.10.2) for every kernel
  - Pallas v6e lowering on CPU, which abstains on everything in this setup
- **`ledger*.jsonl`:** per-call cost.
  - Total **$346.45** at list price, covered by AWS credits.
  - That includes $28.60 of results lost to an early bug: the generator held results in memory, and that batch was regenerated.

## TPU verdicts (`tpu/tpu_verdicts.jsonl`)
- **Coverage:** step-1 survivors judged on v6e-1 VMs in 3 batches, round 1 (329), round 2 (1,030) and a mop-up (130, running when
  this was pushed), each 8 or 3 VMs in parallel (`tpu/fleet_run*.sh`).
- **Crash handling:** `tpu/crashloop.py` sets aside a kernel that crashes the TPU compiler and judges it alone.
- **As of the push (r1 + r2, 1,215 judged):**
  - **344 pass**, 209 correct_slow, so **553 correct** (50% of 1,103 judged non-harness)
  - failures: 394 compile, 61 mosaic, 36 wrong, 35 vmem, 24 runtime
  - 112 harness_oom (inputs too large for one chip; not the kernel's fault)

## Before training or publishing
- **Filters:** run the team's contamination filters (layers 1-2, near-dup, val-family rule) and the coverage check.
- **Terms:** the Kimi K3.1-Pro licence places no restriction on outputs. These outputs were generated through Amazon Bedrock, whose
  Service Terms (section 50.5) address using an AI service to generate content for training models. Check both before redistributing.
- **Next:** repair rounds 2-3 (failed kernel + the oracle's feedback → Kimi's fix) are in progress.
