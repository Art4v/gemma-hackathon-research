"""Structure seeds: declared per-task transforms of the integer and structural inputs that seeded()'s fold_in never reaches.

Lebron James (issue #1, 6704387016; accepted by clanker 6704395827): inputs a create_inputs builds without the PRNG are the same on every
seed, so a kernel can be right only for them. 6p's page table is arange (every sequence's pages contiguous and in order; 3p's is
the same), so a kernel that DMAs a contiguous run of pages passed at 8x XLA; 11p's group_sizes is full(M // G), so a batched dense
matmul passed as a grouped GEMM. For seed s >= 1 the oracle applies the task's transform to the inputs after create_inputs; seed 0
stays the task as written. The reference runs on the same transformed inputs, so correctness is still kernel against reference,
and the rule "correct on every seed" now covers the structure too. The same transform for every arm; rows record structure_seed
and the description below.

Each transform keeps every shape and dtype (the kernel is compiled once, at seed 0's abstract inputs) and stays inside what the
task's own reference computes correctly. tests/test_tpu_oracle.py StructureValidity checks both on the real references at toy size.

Constant floating inputs (clanker 6704617531, guards Lebron James 6704625153), the same for every task and arm, applied after the
task's own transform above: norm weights and biases built with jnp.ones / jnp.zeros are the same on every seed, so a kernel that
skips the affine step (or ignores 17p's all-ones mask) matched the reference on every seed. For seed s >= 1, a floating array leaf
with at least 2 elements, all equal to c (constant_floats), is redrawn: c * U[0.5, 1.5] for c != 0 (so 1 -> U[0.5, 1.5], sign and
positivity kept: every variance stays > 0), U[-0.5, 0.5] for c = 0, and a declared mask (MASKS) becomes a random 0/1 mask, P(1) = 0.9,
with the diagonal kept 1 so every row keeps a 1. Size-1 leaves are scalar hyperparameters (eps, a (1,) scale) and stay. The oracle
picks the leaves once per task on seed 1's inputs and drops (and records) any whose draw makes the reference's output non-finite,
so every seed and kernel of the task uses the same leaves; rows record them as float_leaves (and float_leaves_skipped).
Checked per reference (FloatValidity runs 22k, 34k, 43k and 17p at toy size): the varied leaves are norm weights and biases,
batch-norm running means, running variances that are ones (43k, 48k: U[0.5, 1.5], positive under sqrt(var + eps)), a conv bias
(28k), and 17p's pair mask (multiplied into the activations: any 0/1 mask is valid). Two draws change nothing: 26k's in_weight
multiplies an instance norm over one element (always 0), and 28k's conv_bias is constant along the axis its LayerNorm subtracts.

Tasks whose seeds >= 1 differ from JAXBench as written, beyond fold_in's new PRNG draws: 3p, 6p, 7p, 11p, 51p-54p, 58p, 62p, 63p (above); 17p (mask); 22k, 25k,
26k, 28k, 29k, 31k, 34k, 40k, 43k, 45k, 46k, 48k (constant float leaves), as listed by oracle/int_inputs.py's const_float_leaves.
Imports without JAX; jax is imported inside the transforms.
"""
import functools

SALT = 0x57C7  # the structure draws' own key: independent of create_inputs' keys, the same for every arm
FLOAT_SALT = 0xF1  # folded into the seed's structure key, then the leaf index, for the constant-float draws
MASKS = {"17p_Triangle_Multiplication": (1,)}  # pair mask (N, N, 1), as written all ones; the reference computes pair_act * mask
MASK_P = 0.9


def _paged(xs, key, pages: tuple, lens_at: int, table_at: int):
    """A paged-KV task: xs[pages] are page pools (physical page on axis 0), xs[lens_at] the per-sequence kv lengths, xs[table_at] the
    page table. Physical page p moves to slot perm[p] and the table follows (perm[table]; negative entries, padding, stay), so the
    table's pages are a random draw over all physical pages and every sequence still reads its own logical KV: the reference's
    answer is unchanged by the relabelling alone. kv_lens is drawn uniform in [1, kv_lens] per sequence: the references mask keys at
    position >= kv_len, so a kernel ignoring kv_lens is wrong. Query offsets (cu_q_lens) stay: the references read one decode token
    per sequence from them."""
    import jax, jax.numpy as jnp
    xs, table, lens = list(xs), xs[table_at], xs[lens_at]
    k_perm, k_lens = jax.random.split(key)
    perm = jax.random.permutation(k_perm, xs[pages[0]].shape[0]).astype(table.dtype)
    inv = jnp.argsort(perm)  # new[perm[p]] = old[p]  <=>  new = old[inv]
    for i in pages:
        xs[i] = xs[i][inv]
    xs[table_at] = jnp.where(table >= 0, perm[jnp.maximum(table, 0)], table)
    xs[lens_at] = jax.random.randint(k_lens, lens.shape, 1, lens + 1, lens.dtype)
    return tuple(xs)


def _groups(xs, key):
    """11p (lhs, rhs, group_sizes, max_expert_size). Each group size uniform in [0, max_expert_size], zeros included. The reference
    slices a static max_expert_size window per group and masks by count, so no group may exceed it, and dynamic_slice clamps a window
    that would run past M, so every start + max_expert_size must be <= M; with max_expert_size = M // G both hold for any sizes in
    range. That forces sum(group_sizes) <= M, equal to M only when every group is full: rows past the total are zero in the reference."""
    import jax
    lhs, rhs, sizes, mes = xs
    return lhs, rhs, jax.random.randint(key, sizes.shape, 0, mes + 1, sizes.dtype), mes


def _relabel_pages(xs, key, pages: tuple, table_at: int):
    """A flat page table (51p-54p: jnp.arange(total_pages), so every sequence's pages were contiguous and in order). Physical page p
    moves to slot perm[p] in each pool of xs[pages] and the table follows (perm[table]), so every logical page reads the same data
    and the reference's answer is unchanged by construction; a kernel or reference that assumes the table is arange is wrong.
    Lengths, offsets and top-k indices stay as written: they carry these kernels' own contracts (Lebron James, 6710707491)."""
    import jax, jax.numpy as jnp
    xs, table = list(xs), xs[table_at]
    perm = jax.random.permutation(key, xs[pages[0]].shape[0]).astype(table.dtype)
    inv = jnp.argsort(perm)  # new[perm[p]] = old[p]  <=>  new = old[inv]
    for i in pages:
        xs[i] = xs[i][inv]
    xs[table_at] = jnp.where(table >= 0, perm[jnp.maximum(table, 0)], table)
    return tuple(xs)


def _relabel_slots(xs, key, pages: tuple, table_at: int, slots_at: int):
    """58p (cache, positions, block_table, token_to_req_indices, kv_slot_mapping, rms_weight, cos_sin_cache, rope_cache): one pool
    holds the state pages (through block_table, as written jnp.arange) and the KV pages the boundary tokens are written to (through
    kv_slot_mapping, slot = page * rows + row, -1 for padding), and rope_cache has the same pages. Pages are relabelled as in
    _relabel_pages and each slot keeps its row on its page's new label, so every state page and KV slot holds the same data; the
    output cache moves with the pages on both sides. Positions and token_to_req_indices stay: prefill mode fixes them by
    compress_ratio. Rows per page is the cache's axis 1: the reference addresses both pools as slot // physical_page_size."""
    import jax, jax.numpy as jnp
    rows, slots = xs[pages[0]].shape[1], xs[slots_at]
    perm = jax.random.permutation(key, xs[pages[0]].shape[0]).astype(xs[table_at].dtype)  # _relabel_pages' draw from the same key
    xs = list(_relabel_pages(xs, key, pages, table_at))
    xs[slots_at] = jnp.where(slots >= 0, perm[jnp.maximum(slots, 0) // rows] * rows + slots % rows, slots)
    return tuple(xs)


def _prefill_split(xs, key, qsl_at: int, dist_at: int, lens_at=None, init_at=None):
    """62p, 63p (GDN): query_start_loc is cumsum([0] + [1] * decode + [L] * prefill) padded with its last value, and distribution is
    [num_decode, num_seqs, num_seqs], so the split was always even. The prefill requests' lengths are redrawn as random cut points
    of the same total (each >= 1), so decode rows, padding and the token count stay as written. seq_lens (62p) follows: as written a
    prefill request's seq_len is its length. has_initial_state (63p) is drawn per prefill request: as written all False, so a kernel
    ignoring the carried recurrent state was correct. Google's kernels tile each request by cdiv(len, chunk) and take any length."""
    import jax, jax.numpy as jnp
    xs, qsl, dist = list(xs), xs[qsl_at], xs[dist_at]
    n_dec, n = int(dist[0]), int(dist[1])  # transforms run eagerly on create_inputs' arrays (seed 0's abstract never comes here)
    lo, total, k = qsl[n_dec], int(qsl[n] - qsl[n_dec]), n - n_dec
    if k < 1 or total < k:
        raise ValueError(f"prefill split: {k} requests over {total} tokens")
    k_cut, k_init = jax.random.split(key)
    cuts = jnp.sort(jax.random.choice(k_cut, total - 1, (k - 1,), replace=False) + 1).astype(qsl.dtype)
    ends = jnp.concatenate([cuts, jnp.array([total], qsl.dtype)])
    new_lens = jnp.diff(jnp.concatenate([jnp.zeros((1,), qsl.dtype), ends]))
    xs[qsl_at] = qsl.at[n_dec + 1:n + 1].set(lo + ends)
    if lens_at is not None:
        old = xs[lens_at]
        if not bool(jnp.all(old[n_dec:n] == (qsl[n_dec + 1:n + 1] - qsl[n_dec:n]))):
            raise ValueError("prefill split: seq_lens of the prefill requests are not their lengths")
        xs[lens_at] = old.at[n_dec:n].set(new_lens.astype(old.dtype))
    if init_at is not None:
        xs[init_at] = xs[init_at].at[n_dec:n].set(jax.random.bernoulli(k_init, 0.5, (k,)))
    return tuple(xs)


def _ragged_kv(xs, key, lens_at: int, cu_at: int):
    """7p (queries, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs). kv_lens and cu_q_lens are module constants and only the
    page table is drawn, so a kernel could hard-code the KV lengths. Each kv_len is drawn uniform in [q_len, as written], q_len from
    cu_q_lens: every query stays inside its KV, padded sequences (kv_len 0) stay 0, and pages only cover less. q_lens stay as
    written: their capacities fill max_num_batched_tokens exactly, so shortening them would create padding rows the reference
    defines as zero and production kernels leave unspecified (Lebron James, issue #1 6708956319)."""
    import jax, jax.numpy as jnp
    xs, lens, cu = list(xs), xs[lens_at], xs[cu_at]
    lo = jnp.minimum(cu[1:] - cu[:-1], lens).astype(lens.dtype)
    xs[lens_at] = jax.random.randint(key, lens.shape, lo, lens + 1, lens.dtype)
    return tuple(xs)


_FLAT_DESC = "physical pages relabelled (pool rows permuted, page_indices follows; logical KV unchanged); lengths and top-k as written"
_PAGED_DESC = ("page table permuted over all physical pages with the KV pages moved to match (logical KV unchanged); kv_lens "
               "uniform in [1, as written]")
TRANSFORMS = {
    # (queries, k_pages, v_pages, kv_lens, page_indices[seq, page], cu_q_lens)
    "6p_Paged_Attention": (functools.partial(_paged, pages=(1, 2), lens_at=3, table_at=4), _PAGED_DESC),
    # (ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv, kv_lens, page_indices[seq * pages_per_seq], cu_q_lens, distribution); the
    # reference also writes the new token through the table and returns the updated cache, which moves with the pages
    "3p_MLA_Attention": (functools.partial(_paged, pages=(4,), lens_at=5, table_at=6), _PAGED_DESC),
    # (queries, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs); page_indices is already a seeded permutation
    "7p_Ragged_Paged_Attention": (functools.partial(_ragged_kv, lens_at=2, cu_at=4),
                                  "kv_lens uniform in [q_len, as written] per sequence (q_lens, cu_q_lens and padding unchanged)"),
    # step 5's DeepSeek V4 tasks (level 2 only), each create_inputs' returned tuple: page pools, then the flat table
    **{t: (functools.partial(_relabel_pages, pages=pg, table_at=tb), _FLAT_DESC) for t, (pg, tb) in {
        "51p_DeepSeek_V4_CSA": ((1, 2), 4),               # (q, cache_kv_nope, cache_kv_rope, topk_indices, page_indices, ...)
        "52p_DeepSeek_V4_HCA": ((1,), 4),                 # (q, kernel_cache, kv_lens, kv_lens_to_attend, page_indices, ...)
        "53p_DeepSeek_V4_MLA_SWA": ((2,), 4),             # (q, new_kv, kernel_cache, kv_lens, page_indices, ...)
        "54p_DeepSeek_V4_Masked_Dense_MLA": ((1, 2), 5),  # (q, cache_kv_nope, cache_kv_rope, kv_lens, topk_indices, page_indices, ...)
    }.items()},
    # (cache, positions, block_table, token_to_req_indices, kv_slot_mapping, rms_weight, cos_sin_cache, rope_cache)
    "58p_DeepSeek_V4_KV_Compressor": (functools.partial(_relabel_slots, pages=(0, 7), table_at=2, slots_at=4),
                                      "physical pages relabelled in cache and rope_cache; block_table and kv_slot_mapping follow (same "
                                      "data at every state page and KV slot); positions as written"),
    # 62p (qkv, b, a, conv_state, recurrent_state, conv_weight, conv_bias, a_log, dt_bias, query_start_loc, state_indices, distribution,
    # seq_lens); 63p (mixed_qkv, b, a, recurrent_state, A_log, dt_bias, query_start_loc, state_indices, distribution, has_initial_state)
    "62p_Causal_Conv1d_GDN": (functools.partial(_prefill_split, qsl_at=9, dist_at=11, lens_at=12),
                              "prefill lengths redrawn as random cut points of the same total (each >= 1); seq_lens follows"),
    "63p_GDN": (functools.partial(_prefill_split, qsl_at=6, dist_at=8, init_at=9),
                "prefill lengths redrawn as random cut points of the same total (each >= 1); has_initial_state drawn per prefill request"),
    "11p_Megablox_GMM": (_groups, "group_sizes uniform in [0, max_expert_size] per expert (uneven, zeros allowed, sum <= M; rows "
                                  "past the sum are zero)"),
}


def describe(task: str) -> str | None:
    t = TRANSFORMS.get(task)
    return t[1] if t else None


def describe_floats(task: str, varied: list) -> str | None:
    """The constant-float draws for `varied` ([{leaf, value}]), in words, or None when there are none."""
    def one(f):
        c = f["value"]
        if f["leaf"] in MASKS.get(task, ()):
            return f"{f['leaf']} (mask all {c:g} -> random 0/1, P(1) = {MASK_P}, diagonal kept 1)"
        return f"{f['leaf']} (all {c:g} -> U[{min(.5 * c, 1.5 * c):g}, {max(.5 * c, 1.5 * c):g}])" if c else f"{f['leaf']} (all 0 -> U[-0.5, 0.5])"
    return "constant float leaves redrawn: " + ", ".join(map(one, varied)) if varied else None


def constant_floats(inputs) -> list:
    """Leaf indices of the floating array leaves with at least 2 elements, all equal. Reads the values: concrete arrays only."""
    import jax, jax.numpy as jnp
    return [i for i, x in enumerate(jax.tree_util.tree_leaves(inputs)) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
            and x.size >= 2 and bool(jnp.all(x == x.reshape(-1)[0]))]


def _redraw(x, key, mask: bool):
    import jax, jax.numpy as jnp
    if mask:
        eye = jnp.arange(x.shape[0])[:, None] == jnp.arange(x.shape[1])[None, :]
        return (jax.random.bernoulli(key, MASK_P, x.shape) | eye.reshape(eye.shape + (1,) * (x.ndim - 2))).astype(x.dtype)
    c = float(x.reshape(-1)[0])
    u = jax.random.uniform(key, x.shape, jnp.float32, 0.5, 1.5)
    return (c * u if c else u - 1.0).astype(x.dtype)


def apply(task: str, inputs, seed: int, floats=()):
    """The inputs seed `seed` runs on: create_inputs' own for seed 0, else the task's transform (if any) with a key derived from SALT
    and the seed, then the constant-float draw for each leaf index in `floats` (the caller picks them with constant_floats)."""
    t = TRANSFORMS.get(task)
    if not seed or (t is None and not floats):
        return inputs
    import jax
    key = jax.random.fold_in(jax.random.key(SALT), seed)
    xs = type(inputs)(t[0](tuple(inputs), key)) if t else inputs
    if not floats:
        return xs
    leaves, tree = jax.tree_util.tree_flatten(xs)
    fk = jax.random.fold_in(key, FLOAT_SALT)
    return jax.tree_util.tree_unflatten(tree, [_redraw(x, jax.random.fold_in(fk, i), i in MASKS.get(task, ())) if i in floats else x
                                               for i, x in enumerate(leaves)])
