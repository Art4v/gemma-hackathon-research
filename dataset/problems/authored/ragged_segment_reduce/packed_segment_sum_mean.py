"""Packed ragged segment sum + segment mean over concatenated sequences (lengths -> segment ids -> f32 segment_sum -> divide by length)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'packed_segment_sum_mean',
    'total_tokens': 32768,
    'hidden': 4096,
    'num_segments': 64,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_cut = jax.random.split(key, 2)
    T = CONFIG['total_tokens']
    D = CONFIG['hidden']
    B = CONFIG['num_segments']
    x = jax.random.normal(k_x, (T, D), dtype=dtype)
    # B-1 random cut points in [0, T]; sorted diffs give lengths summing exactly to T.
    cuts = jax.random.randint(k_cut, (B - 1,), 0, T + 1, dtype=jnp.int32)
    cuts = jnp.sort(cuts)
    bounds = jnp.concatenate([
        jnp.zeros((1,), jnp.int32), cuts, jnp.full((1,), T, jnp.int32)
    ])
    lengths = (bounds[1:] - bounds[:-1]).astype(jnp.int32)
    return x, lengths


def workload(x, lengths):
    T = x.shape[0]
    B = lengths.shape[0]
    segment_ids = jnp.repeat(
        jnp.arange(B, dtype=jnp.int32), lengths, total_repeat_length=T
    )
    seg_sum = jax.ops.segment_sum(
        x.astype(jnp.float32), segment_ids, num_segments=B,
        indices_are_sorted=True,
    )
    denom = jnp.maximum(lengths, 1).astype(jnp.float32)[:, None]
    seg_mean = seg_sum / denom
    return seg_sum.astype(x.dtype), seg_mean.astype(x.dtype)
