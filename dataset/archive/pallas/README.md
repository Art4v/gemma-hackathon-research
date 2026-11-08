# Pallas generation pipeline (paused Nov 8)

Kept as it was when kernel generation was paused. Not used by the dataset, not run by its tests, and not imported by anything in
`dataset/scripts/`.

| File | Role |
|---|---|
| `scripts/generate.py` | constructor agent writes Pallas kernels per round (first attempt, then repairs from oracle feedback) |
| `scripts/prescreen.py` | parse check and the oracle's `--static-only` kernel check before submission |
| `scripts/oracle_submit.py`, `scripts/oracle_collect.py` | build oracle queue jobs; collect verdicts into trajectories |
| `scripts/assemble.py`, `scripts/verify_dataset.py` | turn verified trajectories into single-turn and repair training examples |
| `scripts/render_chat.py` | the chat rendering of problems, kernels and oracle feedback |
| `scripts/constructor/prompts.py` | prompts-1, including the kernel-writing system prompt |
| `tests/` | their tests and synthetic fixtures |

To revive: move the scripts back to `dataset/scripts/`, restore `render_chat` imports, and follow the archived design in git history.
Imports reference `dataset/scripts/common.py`; some names there (for example `candidate_id`, the candidate and verdict schemas) are
still present for this reason.
