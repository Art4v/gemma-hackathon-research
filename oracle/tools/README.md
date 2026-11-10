# Karthik's TPU job tools (as used Nov 9-10)

- `ops_loop.sh JOB...`: run packed jobs back to back on one TPU, fetch, `oracle/collect.py`, publish `oracle-results/<job>`.
  `RUNNER=oracle/vm.sh` (GCP; needs `ZONE`, `NAME`) or `RUNNER=oracle/colab.sh` (Colab; `NAME` = session name). It handles both
  status formats, a hang guard (kills a run whose rows are all written but whose process lingers), and refuses to publish 0 rows.
- `crashloop.py BASE ITEMS.jsonl`: one round to completion despite XLA compiler segfaults (a crash kills the oracle process). Parts
  run the unjudged items; on a short stop the next unjudged item is the suspect, set aside; suspects then run alone. Crashing alone
  too = `harness_crash` (two crashes). Uses `ops_loop.sh` and `make_pack.py`.

Both assume this layout: the scripts sit next to a `repo/` checkout (`D/repo`). Edit `D`/`S` at the top for yours.
Collect a split round by merging the parts' results into one job dir under the round's canonical job id (e.g. `dataset-m1-r1`), packed
with only the judged items, then `oracle_collect.py <job> --jobs-dir <dir> --results-dir <dir>`: split job ids confuse its batch parsing.
