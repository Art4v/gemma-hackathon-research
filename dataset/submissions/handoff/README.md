# TPU handoff to Aarav (Nov 10)

Karthik's GCP no longer runs oracle jobs; from here on, every TPU job runs on Aarav's GCP credits. These three item files are
ready, every item checked. Each is a make_pack.py input (`id`, `task`, `ref_code`, `code`, `model`), oracle sha c9a07d9298a1.

| file | items | what |
|---|---|---|
| dataset-prod-r1.items.jsonl | 22 | production problems, round 1 (all bf16; based_chunk excluded) |
| dataset-m1-r2h.items.jsonl | 6 | Hruday's m1 round-2 repairs (the 6 open trajectories) |
| dataset-kbk-r1w3.items.jsonl | 43 | Hruday's KernelBook round-1 wave 2 |

To run one, use the oracle as usual: `oracle/make_pack.py ITEMS oracle/jobs/JOB --job-id JOB ...`, then run it on a v6e.
`oracle/tools/crashloop.py` (with `ops_loop.sh`) splits a job at compiler segfaults and judges each suspect alone. Wave 1 hit three
segfaults in 248 items, and one kernel (NonLocalBlock_8363) crashed twice. Please publish results as `oracle-results/<JOB>` so
Karthik's agent can collect them into the canonical jobs (`dataset-kbk-r1`, `dataset-m1-r2`, `dataset-prod-r1`).

The KernelBook round-1 repairs (r2/r3) that follow will also come as item files, to this folder.
