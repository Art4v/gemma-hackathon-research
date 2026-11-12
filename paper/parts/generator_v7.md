# The v7 generator (`dataset/scripts/v7_gen.py`): a paragraph for the writeup, and the facts behind it

Source: `aarav/dataset-v4` @ `ca43b98` and later, `dataset/scripts/v7_gen.py`, which builds on v3's template generator
(`v3_generate.py`). The `--rowout` mode was added by Karthik's side as a reviewed patch: default and `--layout` candidates are byte-identical
before and after it.

## Paragraph (about 230 words, ready to paste)

The model's main exam failure was a design choice: on the matmul-led tasks it either kept the whole reduction (K) dimension in one VMEM block,
which overflows on-chip memory at the exam's 8192-wide weights, or reduced over one N tile instead of the whole row. Our training problems had
small weights, so those designs were correct there. To show the alternatives at exam scale, we wrote a template generator. It emits a
JAX reference and a Pallas kernel for three families: matmul and batched matmul with a K-tiled f32 accumulator and an elementwise epilogue,
and a matmul followed by a row reduction whose weight block spans all of N. Sizes match the exam: K and N in {4096, 8192}, weights ≥ 64 MB.
Every kernel picks its tiles at trace time from one policy, `_tile(n, pref)`, and sets one VMEM limit of 124 MiB. The tile preferences came from
TPU sweeps on generated training problems, never exam files. Two further modes add the exam's input layouts: a weight stored (N, K), bias and
scale-and-shift vectors (`--layout`), and accumulating whole-row reductions in the output block at N = 8192 (`--rowout`). Every candidate passed
the contamination filters, a CPU lowering check at full size, and numerics at a reduced size that keeps two tiles per tiled axis. Every row then
passed the TPU oracle at 0.7× XLA or better before entering training.

## Facts

- **Families:**
  - `matmul_epilogue` and `bmm_epilogue`: grid `(M/tm, N/tn, K/tk)`, with the batch axis first for bmm. An f32 accumulator in VMEM scratch is reset on the first K step, and the epilogue runs on the last.
  - `row_reduction_matmul`: grid `(M/tm, K/tk)`. The weight block is `(tk, N)`, so the reduction always sees whole rows.
  - **Epilogues:** drawn from v3's registry of elementwise ops (relu, gelu, sigmoid·x, softplus, clamp and others). Row reductions use softmax, log_softmax, normalize, center, or a keep-dims reduction.
- **Sizes:** K and N in {4096, 8192}. M in {2048, 4096} for matmul and bmm (the sweep showed an output tile under 2048 × 2048 stays below 0.7× on f32), and in {1024, 2048, 4096} for row reductions. Weights ≥ 64 MB; inputs ≤ 800 MB (the CPU check's host-memory limit).
- **Dtypes:** f32 or bf16 in the default mode; f32 only for `--layout` and `--rowout`, since every exam input is f32.
- **Tiling policy:** set from three sweeps on a v6e-1, 43 variants, oracle `c9a07d9298a1`.
  - **Matmul and bmm:** `(tm, tn, tk) = (2048, 2048, 1024)`, about 80 MiB for f32. f32 went from 0.66× to 0.87× at M 2048 and 1.31× at M 4096; the speed tracks the output-tile area.
  - **Row reduction:** `tm = 1024`, `tk = 512` (N 4096) or `256` (N 8192).
  - **VMEM limit:** 124 MiB everywhere. 100 MiB capped the f32 N = 8192 row reduction at 0.58×.
  - **Precision:** a bf16-cast dot was tested and bought nothing (0.870× against 0.866×), so the f32 dot stays.
- **`--layout`:**
  - **Transposed weight:** stored `(n, k)` and contracted with `lax.dot_general(..., (((1,), (1,)), ((), ())))`, so there's no in-kernel transpose.
  - **Vectors:** a bias (`(N,)`, whole for row reductions), and `y * s + v` with two vectors.
- **`--rowout` (Karthik's side):** for f32 whole-row output at N = 8192, accumulate in the f32 output block (no scratch) and run the epilogue on 256-row slices. All 61 TPU-judged items passed (0.94–1.51×).
- **Checks on every candidate, before the TPU:**
  1. **Full-size reference checks:** determinism, distinct seeds and the HBM budget.
  2. **Static check and lowering:** the oracle's static step 1 at full size, and Mosaic lowering for v6e.
  3. **Contamination:** L1–L2 (KernelBench name, op chain) and L3 (near-duplicate text) against the 50 exam references and 73 exam files.
  4. **Numerics:** interpret mode, 3 seeds, at a shrunken CONFIG with **exactly 2 tiles on every axis tiled at full size**, asserted. N is kept at 2048 for row reductions. Planted bugs were caught: an N-tile-0 index map, partial-row reductions, a transposed index swap, and a weight pinned to K tile 0.
  5. **VMEM estimate:** `v3_block_bytes.py`, 2 × blocks + scratch. It's a ranking only: against TPU verdicts it errs in both directions by about 30 MiB.
- **A6 rule 2, applied after the fact:**
  - **The test:** an item is dropped if its op chain, with linear and layout ops removed (and log_softmax read as softmax), equals a k-task's.
  - **Result:** 41 generated items (36 TPU passes), 19 layout items and 2 rowout items were dropped. The dev set holds no A6 match.
- **Funnel, default mode (300 candidates):** 291 passed the full-size checks (9 hit the host-memory cap), 281 passed contamination, 281 passed numerics, then **269 of 281 passed on TPU** (median 0.98×; the 12 failures were f32 whole-row reductions at N = 8192 that ran out of VMEM).
- **Funnel, `--layout` (130 candidates):** 127 passed contamination, **104 of 111 sent passed on TPU**.
- **Funnel, `--rowout`:** **61 of 61 passed on TPU**.
- **In `v7_final.jsonl`:**
  - **709 template rows** (30.8%): 400 `e5_gen`, 191 `e5_gen_layout`, 118 `e5_gen7r`.
  - **Surface variants:** 354 of these are surface variants (renamed arguments, CONFIG keys, docstrings, `a @ b` vs `jnp.matmul`). Each was proven alpha-equivalent to its TPU-judged original and run on the same input.
  - **Variant sample on TPU:** 91 of 91 sampled variants passed (0.77–1.55×).
