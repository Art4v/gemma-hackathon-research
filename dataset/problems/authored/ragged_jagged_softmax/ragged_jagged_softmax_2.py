"""Segmented softmax over variable-length jagged segments per packed score row."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ragged_jagged_softmax',
    'num_rows': 4096,
    'total_len': 16384,
    'num_segments': 64,
    'min_seg_weight': 64,
    'max_seg_weight': 512,
    'scale': 0.125,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_scores, k_segs = jax.random.split(key, 2)
    num_rows = CONFIG['num_rows']
    total_len = CONFIG['total_len']
    num_segments = CONFIG['num_segments']

    scores = jax.random.normal(k_scores, (num_rows, total_len), dtype=dtype)

    raw_weights = jax.random.randint(
        k_segs,
        (num_rows, num_segments),
        minval=CONFIG['min_seg_weight'],
        maxval=CONFIG['max_seg_weight'],
        dtype=jnp.int32,
    )
    lengths = (raw_weights * total_len) // jnp.sum(raw_weights, axis=-1, keepdims=True)
    rem = total_len - jnp.sum(lengths, axis=-1, keepdims=True)
    lengths = lengths.at[:, -1:].add(rem)
    segment_offsets = jnp.pad(jnp.cumsum(lengths, axis=-1), ((0, 0), (1, 0)))

    return scores, segment_offsets


def workload(scores, segment_offsets):
    total_len = scores.shape[-1]
    num_segments = CONFIG['num_segments']
    scale = CONFIG['scale']
    positions = jnp.arange(total_len, dtype=jnp.int32)

    def row_jagged_softmax(row_scores, row_offsets):
        x = row_scores.astype(jnp.float32) * scale
        seg_ids = jnp.searchsorted(row_offsets[1:], positions, side='right')
        seg_max = jax.ops.segment_max(x, seg_ids, num_segments=num_segments)
        exp_x = jnp.exp(x - seg_max[seg_ids])
        seg_sum = jax.ops.segment_sum(exp_x, seg_ids, num_segments=num_segments)
        out = exp_x / seg_sum[seg_ids]
        return out.astype(row_scores.dtype)

    return jax.vmap(row_jagged_softmax)(scores, segment_offsets)
