| Arm | pass within 3 (JAXBench 50) | correct within 3 | pass@1 | parity within 3 | k-33 pass | p-17 pass | extended-58 pass | all-64 pass |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 E4B (base), reference only | 0/50 | 0/50 | 0/50 | 0/50 | 0/33 | 0/17 | 0/58 | 0/64 |
| Gemma 4 E4B (base) + 25k tokens of Pallas docs | 0/50 | 0/50 | 0/50 | 0/50 | 0/33 | 0/17 | 0/58 | 0/64 |
| Gemini Flash + docs | 7/50 | 36/50 | 1/50 | 4/50 | 5/33 | 2/17 | 7/58 | 8/64 |
| Gemma 4 E4B + our LoRA | 9/50 | 9/50 | 7/50 | 9/50 | 8/33 | 1/17 | 9/58 | 9/64 |

| ours vs | pass within 3: ours-only / other-only, p | correct within 3: ours-only / other-only, p |
|---|---|---|
| bare | 9 / 0, p = 0.0039 | 9 / 0, p = 0.0039 |
| docs | 9 / 0, p = 0.0039 | 9 / 0, p = 0.0039 |
| flash_docs | 8 / 6, p = 0.7905 | 1 / 28, p = 0.0 |
