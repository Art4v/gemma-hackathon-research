# TPU verdicts for the exam-scale generated items (v7)

Run Nov 11-12 2026 on two on-demand v6e-1 VMs on Karthik's GCP project. Oracle sha `c9a07d9298a1`
(`karthik/kbk` @ `86c31a9`), 3 seeds, speed bar 0.7.

- `gen7-full2.merged.jsonl`: one row per item for all 281 items in `dataset/v7_gen/items.jsonl` on
  `aarav/dataset-v4` @ `dd063b8` (sha256 prefix `6dae1a3d682e`). 269 pass, 12 vmem. This is the file
  to build training rows from.
- `gen7-full2.jsonl`, `gen7-tail45.jsonl`, `gen7-full2-rev.jsonl`: the raw per-VM results the merge
  came from. 176 items were judged on both VMs and every verdict agrees.
- `gen7-pilot-s1.jsonl`: the 6-item pilot on the first generation (`c7e23d6`), before the tile sweep.
- `gen7-sweep{1,2,3}.jsonl` with `sweep*_items.jsonl`: the tile sweeps (43 variants) that chose the
  prefs `(2048, 2048, 1024)` for matmul and the 124 MiB limit.

The first VM rebooted by itself at 23:14 UTC; the job resumed from its shard file, so its results
file holds rows from before and after the reboot.
