"""Jagged segment softmax: scale -> per-segment max -> exp -> per-segment sum -> normalize over packed variable-length segments."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'jagged_segment_softmax',
    'batch': 16,
    'total_len': 2097152,
    'num_segments': 1024,
    'scale': 0.125,
    'score_std': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    T = CONFIG['total_len']
    S = CONFIG['num_segments']
    key = jax.random.key(0)
    k_scores, k_cuts = jax.random.split(key, 2)
    scores = (jax.random.normal(k_scores, (B, T), jnp.float32) * CONFIG['score_std']).astype(dtype)
    cuts = jax.random.randint(k_cuts, (B, S - 1), 0, T + 1, dtype=jnp.int32)
    cuts = jnp.sort(cuts, axis=-1)
    offsets = jnp.concatenate(
        [jnp.zeros((B, 1), jnp.int32), cuts, jnp.full((B, 1), T, jnp.int32)], axis=-1
    )
    return scores, offsets


def workload(scores, offsets):
    B, T = scores.shape
    S = offsets.shape[1] - 1
    scale = CONFIG['scale']

    # Segment id per packed position: count interior boundaries <= position.
    interior = offsets[:, 1:-1]  # (B, S-1), values in [0, T]
    rows = jnp.broadcast_to(jnp.arange(B, dtype=jnp.int32)[:, None], interior.shape)
    marks = jnp.zeros((B, T + 1), jnp.int32).at[rows, interior].add(1)
    seg = jnp.cumsum(marks, axis=-1)[:, :T]  # (B, T) in [0, S-1]
    flat_ids = (seg + jnp.arange(B, dtype=jnp.int32)[:, None] * S).reshape(-1)

    x = scores.astype(jnp.float32).reshape(-1) * scale
    n = B * S
    seg_max = jax.ops.segment_max(x, flat_ids, num_segments=n, indices_are_sorted=True)
    seg_max = jnp.where(jnp.isfinite(seg_max), seg_max, 0.0)
    e = jnp.exp(x - seg_max[flat_ids])
    seg_sum = jax.ops.segment_sum(e, flat_ids, num_segments=n, indices_are_sorted=True)
    seg_sum = jnp.where(seg_sum > 0, seg_sum, 1.0)
    out = e / seg_sum[flat_ids]
    return out.reshape(B, T).astype(scores.dtype)
