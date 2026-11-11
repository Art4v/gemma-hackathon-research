# v4 results: JAXBench (TPU oracle, per stage) and val (CPU gates)

Compiled Nov 10, 2026 (Hruday's agent). Every number was computed from the raw files by one agent and then independently recomputed by an
adversarial verifier. All three verifications agreed on every count; their corrections are applied below. Analysis scripts:
`scratchpad/analysis/` (not in git).

## 1. Headline (pre-registered metric `pass_within_3`, 58-task primary set)

Setup for every arm: Modal A10G, Ollama 0.40.14, bare prompt, greedy (temperature 0), 3 turns (turn 0 + 2 repairs with the oracle's
feedback). Oracle `c9a07d9298a1` on TPU v6e (JAX 0.10.2), recorded on all 958 judged rows. v4 = `gemma4-e4b-lora:v4final-e2`, digest
`0f764d3f6798` on every row.

| Arm | `pass_within_3` (primary-58) | correct within 3 (primary-58) | S-p (25 verified p-tasks) |
|---|---|---|---|
| base, bare | 0/58 | 0/58 | 0/25 |
| base, docs | 0/58 | 0/58 | 0/25 |
| v1 LoRA | 0/58 | 0/58 | 0/25 |
| pilot25 LoRA | 0/58 | 0/58 | 0/25 |
| **v4 LoRA** | **1/58** (`12p_RMSNorm`, 1.40x XLA, turn 0) | **3/58** (+ `8p_GEMM` 0.34x, `48k` 0.16x) | **1/25** |

**Significance:** none. Exact two-sided McNemar (the protocol's test): 1 discordant task, p = 1.0. p < 0.05 needs at least 6 discordant
tasks with none the other way. S-p: p = 1.0. Correct within 3: 3 vs 0, p = 0.25.

## 2. The oracle funnel, per stage

Stages (`oracle/tpu_oracle.py`):
- (a) the reply has a code block
- (b) passes oracle step 1, the kernel check (imports, traces, has a `pallas_call` the output depends on, nothing heavy outside). This
  runs on CPU, and everything that passes is judged on the TPU.
- (c) compiles and runs on the TPU on 3 seeds
- (d) correct on all 3 seeds vs a float32 "highest" reference
- (e) pass, meaning speed_vs_xla >= 0.70

Every step-2 failure in every arm is `compile` or `vmem`. No `harness_oom`, `too_large_for_chip`, `timeout`, `runtime`, `mosaic` or
`crash` verdict appears anywhere.

**Tasks reaching each stage within 3 turns, as a/b/c/d/e:**

| Arm | all-64 | primary-58 | S-p-25 | k-33 | p-31 |
|---|---|---|---|---|---|
| base bare | 59/0/0/0/0 | 53/0/0/0/0 | 24/0/0/0/0 | 29/0/0/0/0 | 30/0/0/0/0 |
| base docs | 63/6/0/0/0 | 57/6/0/0/0 | 24/3/0/0/0 | 33/3/0/0/0 | 30/3/0/0/0 |
| v1 | 61/0/0/0/0 | 55/0/0/0/0 | 22/0/0/0/0 | 33/0/0/0/0 | 28/0/0/0/0 |
| pilot25 | 62/0/0/0/0 | 56/0/0/0/0 | 23/0/0/0/0 | 33/0/0/0/0 | 29/0/0/0/0 |
| **v4** | **63/24/7/3/1** | **57/23/7/3/1** | **24/5/2/2/1** | **33/18/5/1/0** | **30/6/2/2/1** |

Wilson 95% intervals for v4 on primary-58: (b) 23/58 28.1–52.5%, (c) 7/58 6.0–22.9%, (d) 3/58 1.8–14.1%, (e) 1/58 0.3–9.1%. The
base docs arm's 6 step-1 survivors were all rejected on the TPU (6 `compile`, 5 `vmem` over their attempts).

**v4 attempts per turn, as a/b/c/d/e, with verdicts:**

| Turn | Attempts | a/b/c/d/e | CPU: no code / compile / not_a_kernel | TPU: compile / vmem / wrong / correct_slow / pass |
|---|---|---|---|---|
| t0 | 64 | 56/24/7/3/1 | 8 / 32 / 0 | 1 / 16 / 4 / 2 / 1 |
| t1 | 63 | 60/23/6/1/0 | 3 / 36 / 1 | 1 / 16 / 5 / 1 / 0 |
| t2 | 63 | 62/23/6/1/0 | 1 / 38 / 1 | 1 / 16 / 5 / 1 / 0 |
| all | 190 | 178/70/19/5/1 | 12 / 106 / 2 | 3 / 48 / 14 / 4 / 1 |

**v4 tasks past step 2:**
- `12p_RMSNorm`: `pass` at turn 0, 1.402x XLA.
- `8p_GEMM`: `correct_slow` at 0.20x, then 0.34x, then 0.34x.
- `48k_Matmul_BatchNorm_BiasAdd_Divide_Swish`: `correct_slow` at 0.16x on turn 0, then `wrong` on turns 1 and 2, because the repair
  broke it.
- `wrong` on every turn: 21k, 24k, 36k, 42k. The oracle records a speed for these too, but those speeds are meaningless and must not be
  quoted.

**The other v4 tasks that passed step 1:**
- `vmem` on all 3 turns, 16 tasks: 13p, 14p, 19k, 20k, 26k, 27k, 29k, 30k, 37k, 38k, 41k, 44k, 47k, 49k, 50k and 64p.
- `compile` on all 3 turns: 57p.
- Several `vmem` failures are 65-66 MB against the kernel's own `vmem_limit_bytes` of 64 MB. A v6e has 128 MiB of VMEM.

**Repairs.** From stage (b) onward, v4's within-3 counts equal its turn-0 counts. Only stage (a) improved, from 52 to 57 tasks on
primary-58.
- 39 of 63 turn-1 codes are byte-identical to turn 0. Base bare does this on 2 of 64, v1 on 3 of 64.
- 54 of 63 turn-2 codes are identical to turn 1, ignoring whitespace.
- v4's training data does include repair rows: 241 real oracle-feedback repairs and 999 injected-bug repairs, 42% of all rows. The
  likely cause is that most of those repairs are injected-bug undos (Karthik's analysis on #5).

**No code block.** Replies without code: base bare 25/192, base docs 12/192, v1 12/192, pilot25 27/192, v4 12/190. In every arm, a
reply lacks code exactly when it was cut off: at the 8,192-token cap, or, once, v4's `repeat_limit` on 62p t0. 62p is v4's only task
with no code on any turn.

## 3. Significance per stage (exploratory: the protocol pre-specifies no test for the funnel)

Exact two-sided McNemar on paired task outcomes, primary-58. Holm correction over the 36-test family, which was built after the data were
seen.

| Stage | v4 vs base bare (v4-only / bare-only) | p | Holm-36 | v4 vs base docs | p |
|---|---|---|---|---|---|
| (b) passed step 1 | 23 / 0 | 2.4e-7 | 8.6e-6 | 20 / 3 | 4.9e-4 (Holm 0.015) |
| (c) compiled and ran | 7 / 0 | 0.0156 | 0.47 | 7 / 0 | 0.0156 |
| (d) correct | 3 / 0 | 0.25 | 1 | 3 / 0 | 0.25 |
| (e) pass | 1 / 0 | 1.0 | 1 | 1 / 0 | 1.0 |

- v4 vs v1 is identical to v4 vs base bare, since both controls are 0 from stage (b) on. Turn 0 alone gives the same tables.
- Paired bootstrap (10,000 resamples), stage (b) v4 minus bare: about +28 to +53 points. The bounds are accurate to about ±1/58.
- The p-task subset (= S-p-25) at stage (b) is 5/25 vs 0, p = 0.0625, not significant.

**Reading.** v4 writes structurally valid Pallas kernels far more often than any control, including base with the docs in its prompt.
That effect is large and robust to multiple-testing correction. Three caveats bound it:
1. **Adapter or stack.** `bare_stack` was not run. v4's served GGUF re-stores its 258 LoRA-target tensors as Q8_0, so the protocol's
   assumption that `bare_stack` equals `bare` does not hold for v4. Base runs on other stacks also had 0 attempts pass step 1, which makes
   a stack effect unlikely, but it is untested.
2. **One task family.** All 18 k tasks that pass step 1 are Matmul/Gemm/BMM-led (18 of 25 such tasks), and none of the 8 Conv-led k
   tasks do. The kbk training data had an algorithm-level contamination review (commit `9ec0e73`): 195 of 941 problems were flagged as
   close to a JAXBench task, 84 were excluded, and the remaining 111 were kept with reasons.
3. **Adapter selection.** Three adapters (pilot25, v1, v4) have been evaluated on JAXBench, so a selection factor of 3 applies.
   Within-3 vs docs then falls to about 0.045.

Stage (b) says nothing about correctness. Stage (c) is not significant after correction (4 of its 7 tasks are `wrong` on every turn),
and stages (d) and (e) are not significant.

## 4. Val (dataset-v1's 23 held-out problems; CPU only, no val ran on a TPU)

**The headline adapter v4final-e2 has no behavioural val.** Its tags were created after the last val file was written. Its only val is
training loss: 1.08 to **0.2601** on 24 loss-val rows. Every behavioural val number for the v4 family comes from the rehearsal adapters
**v4qat-e1/e2**, which used the same quantization-aware recipe on an earlier data build (2,806 rows, val loss 0.2671). Those ran on the
Mac's Ollama, while base and the Modal LoRA val ran on Modal, so the comparisons are confounded by backend. The same v1 adapter scored 3/23
on the Mac and 1/23 on Modal.

**Greedy, oracle step 1 only.** Lowering and numerics were never run on greedy outputs.

| Model | n | has code | **passed step 1** |
|---|---|---|---|
| v4qat-e2 (Mac) | 23 | 20 | **4** (Softplus, Softsign, L1Norm, kb l1 76 conv1d) |
| v4qat-e1 (Mac) | 23 | 9 (14 hit the cap) | 3 |
| v2-q8 (Mac) | 23 | 18 | 4 |
| v1-q8 / v1-f16 (Mac) | 23 | 19 | 3 |
| v1 (Modal) | 23 | 20 | 1 |
| v2 (Modal) | 23 | 22 | 0 |
| v1 bf16 (H100, transformers) | 23 | 19 | 4 |
| **base** (Modal) | 23 | 20 | **0** |

**Sampled (temperature 1.0 / top_p 0.95 / top_k 64, 3 per problem) through three cumulative CPU gates.** The gates are oracle step 1,
then Pallas v6e lowering on CPU, then numerics in TPU interpret mode on CPU. Numerics uses seed 0 only, so it is weaker than oracle step 3.

| Model | per sample: step 1 / lowered / numerics ok | per problem (any of 3): step 1 / lowered / numerics ok |
|---|---|---|
| v4qat-e2 | 10 / 10 / **6** of 69 | 4 / 4 / **3** of 23 |
| v2-q8 (chains 0-2) | 9 / 9 / 5 of 69 | 4 / 4 / 2 of 23 |

Every val numerics pass is on Softplus, Softsign or L1Norm; no authored, kb level-2 or scraped val problem passed. **No val comparison is
significant.** The largest gap is v4qat-e2 vs base at greedy step 1, 4 vs 0, p = 0.125.

## 5. Protocol status (to settle before the writeup)

- The pushed `PROTOCOL.md` (A4) names `v1_r64` as the headline adapter. Amendment **A5** (commit `26e26dc`, Nov 9 16:45 UTC, after the v1
  result and before v4 existed) withdraws that pre-registration and requires reporting **every adapter evaluated on JAXBench, plus the
  total number of evaluations**. A5 is only on the local branch `hruday/protocol-no-headline`. It is not pushed or merged, so v4's status
  as headline depends on it.
- Adapters evaluated on JAXBench so far: pilot25, v1, v4 (all bare, greedy, Modal), plus the base bare and docs arms. The two
  `base_local__mislabelled_*` runs are base-model runs and are reported as such.
- `bare_stack` is still unrun for v4; see section 3.

## 6. v5 (plan-before-code, `v5plans_r64_full`) vs v4final-e2 (added Nov 11)

v5 is the adapter Aarav trained on Karthik's dataset-v5 (`karthik/v5-plans` @ `98f95c3`): v4's rows with a checked `# PLAN` block
above each kernel, trained with v4final's recipe (2 epochs, H100). Adapter sha256 `9fd591c2…`, verified on download.
- **Fold and serving:** folded as v4 (`fold_local.py`, delta at q8_0), GGUF sha256 `de2516a3…`. Served on the same Modal backend as
  `gemma4-e4b-lora:v5full`, digest `641ddde330ed` (`control/modal_v5_create.py`). Every layer except the model equals v4's.
- **Val** (`v5_val/`, 23 held-out problems, CPU gates):
  - greedy step 1: v4 5/23, v5 4/23
  - sampled (69): step 1 9 vs 9; numerics ok 6 vs 6
  - problems numerically ok: 3 vs 3
  - v5 opens every reply with `# PLAN` (23/23, 69/69); v4 never does.
- **JAXBench** (bare, greedy, 3 turns, Modal; 191 rows, all digest `641ddde330ed`): a Modal RemoteError cut turn 2 to 10 rows, and a
  catch-up pass generated the other 53. On primary-58, within 3 turns:

  | Stage | v4 | v5 | only v4 / only v5 | exact McNemar p |
  |---|---|---|---|---|
  | (b) step 1 | 23 | 12 | 13 / 2 | 0.007 |
  | (c) compiled and ran | 7 | 7 | 5 / 5 | 1.0 |
  | (d) correct | 3 | 2 | 2 / 1 | 1.0 |
  | (e) pass | 1 | 1 | 0 / 0 | 1.0 |

  - Both pass only `12p_RMSNorm`. v5 failed it at t0 on VMEM (48 MB against a 32 MB limit) and fixed it at t1 (1.39x XLA). That is
    one real repair.
  - v5 still re-emits its previous code on 48/64 (t0->t1) and 55/63 (t1->t2) repairs; v4's figures are 40/63 and 54/63.
  - **Conclusion (exploratory):** plan-before-code SFT did not improve JAXBench, and it significantly reduced structural validity
    (step 1 down from 23 to 12).
