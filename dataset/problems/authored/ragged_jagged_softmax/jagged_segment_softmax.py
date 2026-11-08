"""Jagged segment softmax: per-head softmax within variable-length segments of packed scores (offsets-delimited)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'jagged_segment_softmax',
    'num_heads': 32,
    'total_len': 1048576,
    'num_segments': 512,
    'head_dim': 128,
    'scale': 1.0 / (128 ** 0.5),
    'score_std': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_scores, k_cuts = jax.random.split(key)
    H = CONFIG['num_heads']
    T = CONFIG['total_len']
    B = CONFIG['num_segments']
    scores = (jax.random.normal(k_scores, (H, T), jnp.float32) * CONFIG['score_std']).astype(dtype)
    cuts = jnp.sort(jax.random.randint(k_cuts, (B - 1,), 0, T + 1, dtype=jnp.int32))
    offsets = jnp.concatenate([
        jnp.zeros((1,), jnp.int32),
        cuts,
        jnp.full((1,), T, jnp.int32),
    ])
    return scores, offsets


def workload(scores, offsets):
    H, T = scores.shape
    B = offsets.shape[0] - 1
    pos = jnp.arange(T, dtype=jnp.int32)
    seg_ids = jnp.searchsorted(offsets, pos, side='right').astype(jnp.int32) - 1
    seg_ids = jnp.clip(seg_ids, 0, B - 1)

    x = scores.astype(jnp.float32).T * CONFIG['scale']  # (T, H)
    seg_max = jax.ops.segment_max(x, seg_ids, num_segments=B, indices_are_sorted=True)  # (B, H)
    seg_max = jnp.where(jnp.isfinite(seg_max), seg_max, 0.0)
    e = jnp.exp(x - seg_max[seg_ids])
    seg_sum = jax.ops.segment_sum(e, seg_ids, num_segments=B, indices_are_sorted=True)  # (B, H)
    seg_sum = jnp.where(seg_sum > 0.0, seg_sum, 1.0)
    out = e / seg_sum[seg_ids]
    return out.T.astype(scores.dtype)
