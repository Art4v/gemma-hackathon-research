# Control benchmark: gemma4:e4b-it-qat on Modal (2026-11-08 06:04)

Verdicts: 384 (373 decided on the CPU at oracle step 1, 11 on the TPU).

| arm | tasks judged | pass@1 | pass within 3 | correct within 3 | parity (>= 0.97x) | median speed of correct |
|---|---|---|---|---|---|---|
| bare | 64 / 64 | 0 | 0 | 0 | 0 | None |
| docs | 64 / 64 | 0 | 0 | 0 | 0 | None |

bare, turn 0 verdicts: {'compile': 62, 'not_a_kernel': 2}
bare, final verdicts: {'compile': 60, 'not_a_kernel': 4}

docs, turn 0 verdicts: {'compile': 55, 'not_a_kernel': 6, 'vmem': 3}
docs, final verdicts: {'compile': 62, 'not_a_kernel': 1, 'vmem': 1}

Level 2 reference check:
```
51p_DeepSeek_V4_CSA                      HOLDS  correct_slow  correct but too slow: 0.63x XLA, need >= 0.70x
52p_DeepSeek_V4_HCA                      FAILS  wrong  wrong output on 3 of 3 seeds; seed 0: max error 1 of max|ref| (XLA default: 2.46e-40); seed 1: max e
53p_DeepSeek_V4_MLA_SWA                  HOLDS  pass  pass: 1.98x XLA
54p_DeepSeek_V4_Masked_Dense_MLA         FAILS  compile  compile error: INTERNAL: Operation not supported at location: loc("sc_mask_generate.6":106:1) The ML
56p_DeepSeek_V4_MHC                      HOLDS  pass  pass: 1.08x XLA
57p_SparseCore_Ragged_Gather             HOLDS  pass  pass: 1.23x XLA
58p_DeepSeek_V4_KV_Compressor            HOLDS  pass  pass: 1.07x XLA
59p_Megablox_GMM_v2                      FAILS  mosaic  Mosaic compile error: INTERNAL: Mosaic failed to compile TPU kernel: Not Implemented: The last dim s
60p_RPAv3                                FAILS  wrong  wrong output on 2 of 3 seeds; seed 0: max error 0.0222 of max|ref| (XLA default: 0); seed 2: max err
61p_Batched_RPA                          HOLDS  pass  pass: 2.51x XLA
62p_Causal_Conv1d_GDN                    HOLDS  pass  pass: 1.69x XLA
63p_GDN                                  HOLDS  pass  pass: 3.54x XLA
64p_Linear_Softmax_Cross_Entropy         FAILS  compile  compile error: No module named 'jaxtyping'
65p_Quantized_Splash_Attention           FAILS  harness_oom  harness error (HBM out of memory or the reference failed), not the kernel: RESOURCE_EXHAUSTED: Ran o
8 of 14 references hold
```
