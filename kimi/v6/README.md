# v6 training set (draft, Nov 11): Kimi K3.1 kernels the TPU oracle verified

- **Format:** v4's exact row format, built with `control/bench.py` `conversation()`: the student's system prompt, the reference turn,
  then the fenced code. Repairs add the failed kernel and the oracle's feedback + "Reply with the corrected Python code block.", as
  JAXBench repair turns do.
- **Targets:** code-only (`thinking: false`), so v6 is evaluated under the same protocol as every other arm. Each row keeps
  `problem_id`, so Kimi's reasoning and plan can be joined back from `kimi/data/out/`.
- **Kinds:**
  - `single`: attempt 1 passed on TPU
  - `correct_slow`: correct but under the 0.7x bar
  - `repair` / `repair_slow`: a failed attempt 1 + oracle feedback, then a verified fix; these come from repair rounds 2-3, still running
- **Never used:** `harness_oom` rows.
- **Draft counts:** 616 rows: 380 `single` + 236 `correct_slow`. Repairs will be added as their TPU verdicts land.
- **Script:** `kimi/scripts/assemble_v6.py`.
- **Before training:** run the team's contamination filters (layers 1-2, near-dup, val-family). Problems were already drawn from the
  filtered pool (see `kimi/README.md`).
