# v7 weight-scaled problems (as of Nov 11)

> Written on Nov 11, before the final build, and covers the weight-scaling step only. The final training file is assembled by
> `assemble_v7.py` and checked by `check_v7.py` (output in `check_v7.out`); the dev slice and base set listed as undecided below were
> settled in that build.

- **`kn_refs.jsonl` / `kn_problems.jsonl`:** 89 weight-scaled variants (`__kn`) of solved, unflagged, matmul-dominant v6-strict problems.
  - Every feature-dimension CONFIG int (in/out features, hidden, ...) is multiplied by one integer factor, so the main matmul reaches max(K, N) of 4096–16384 and a weight of at least 64 MB. Batch ints are divided by an exact power of two if inputs exceed 1.2 GB. The op chain is unchanged.
  - Each variant is shape-checked (`jax.eval_shape`), and K/N are measured from the jaxpr.
  - 33 reach K or N of 8192 or more. 10 are rule B (a reduction over N follows the matmul). 23 more problems were already at exam scale in v6, so no copy was made.
- **`kn_items_existing.jsonl`:** TPU items that re-run each problem's existing verified v6 kernel, unchanged, at the scaled CONFIG (step 2).
- **`kn_log.jsonl`:** per-candidate outcome for all 972 candidates. 687 have no non-batched matmul; the others failed to scale or overshot.
- **Scripts:** `scale_kn_one.py`, `build_kn.py`.
- **Not decided yet:** the dev slice (problem-level holdout) and the base set; see issue #5.
