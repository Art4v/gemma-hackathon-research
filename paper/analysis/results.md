| Arm | pass within 3 (primary-58) | correct within 3 | pass@1 | parity within 3 | k-33 pass | S-p-25 pass | all-64 pass |
|---|---|---|---|---|---|---|---|
| Gemma 4 E4B (base), reference only | 0/58 | 0/58 | 0/58 | 0/58 | 0/33 | 0/25 | 0/64 |
| Gemma 4 E4B (base) + 25k tokens of Pallas docs | 0/58 | 0/58 | 0/58 | 0/58 | 0/33 | 0/25 | 0/64 |
| Gemini Flash + docs | 7/58 | 36/58 | 1/58 | 4/58 | 5/33 | 2/25 | 8/64 |
| LoRA pilot (25 rows) | 0/58 | 0/58 | 0/58 | 0/58 | 0/33 | 0/25 | 0/64 |
| LoRA v1 (dataset-v1, Q4_0 fold) | 0/58 | 0/58 | 0/58 | 0/58 | 0/33 | 0/25 | 0/64 |
| LoRA v4final-e2 (dataset-v4) | 1/58 | 3/58 | 1/58 | 1/58 | 0/33 | 1/25 | 1/64 |
| LoRA v5full (plan-before-code) | 1/58 | 2/58 | 0/58 | 1/58 | 0/33 | 1/25 | 1/64 |
| LoRA v7final (dataset v7, final) | 9/58 | 9/58 | 7/58 | 9/58 | 8/33 | 1/25 | 9/64 |

| v7 vs | pass within 3: v7-only / other-only, p | correct within 3: v7-only / other-only, p |
|---|---|---|
| bare | 9 / 0, p = 0.0039 | 9 / 0, p = 0.0039 |
| docs | 9 / 0, p = 0.0039 | 9 / 0, p = 0.0039 |
| flash_docs | 8 / 6, p = 0.7905 | 1 / 28, p = 0.0 |
| pilot25 | 9 / 0, p = 0.0039 | 9 / 0, p = 0.0039 |
| v1 | 9 / 0, p = 0.0039 | 9 / 0, p = 0.0039 |
| v4 | 8 / 0, p = 0.0078 | 8 / 2, p = 0.1094 |
| v5 | 8 / 0, p = 0.0078 | 7 / 0, p = 0.0156 |
