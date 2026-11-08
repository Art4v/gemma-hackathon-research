"""Jagged segment-wise scaled softmax over packed 1D variable-length segments per row."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'jagged_segment_softmax',
    'num_heads': 32,
    'total_elements': 2097152,
    'num_segments': 4096,
    'scale': 0.125,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    H = CONFIG['num_heads']
    N = CONFIG['total_elements']
    S = CONFIG['num_segments']

    scores = jax.random.normal(k1, (H, N), dtype=dtype)

    raw_lengths = jax.random.randint(k2, (S,), minval=64, maxval=1024, dtype=jnp.int32)
    scaled_lengths = jnp.maximum(1, (raw_lengths * N) // jnp.sum(raw_lengths))
    diff = N - jnp.sum(scaled_lengths)
    scaled_lengths = scaled_lengths.at[-1].add(diff)

    seg_offsets = jnp.concatenate([
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.cumsum(scaled_lengths, dtype=jnp.int32),
    ])
    return (scores, seg_offsets)


def workload(scores, seg_offsets):
    N = scores.shape[-1]
    S = CONFIG['num_segments']
    scale = CONFIG['scale']
    eps = CONFIG['eps']

    token_indices = jnp.arange(N, dtype=jnp.int32)
    seg_ids = jnp.searchsorted(seg_offsets[1:], token_indices, side='right')

    def row_jagged_softmax(row):
        x = row.astype(jnp.float32) * scale
        seg_max = jax.ops.segment_max(x, seg_ids, num_segments=S)
        exp_x = jnp.exp(x - seg_max[seg_ids])
        seg_sum = jax.ops.segment_sum(exp_x, seg_ids, num_segments=S)
        out = exp_x / (seg_sum[seg_ids] + eps)
        return out.astype(scores.dtype)

    return jax.vmap(row_jagged_softmax)(scores)
