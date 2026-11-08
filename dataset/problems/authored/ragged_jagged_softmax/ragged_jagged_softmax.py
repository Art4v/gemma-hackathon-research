"""Scaled jagged softmax over variable-length packed segments per row."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ragged_jagged_softmax',
    'num_rows': 4096,
    'total_cols': 65536,
    'num_segments': 128,
    'min_seg_len': 128,
    'scale': 0.125,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    num_rows = CONFIG['num_rows']
    total_cols = CONFIG['total_cols']
    num_segments = CONFIG['num_segments']
    min_seg_len = CONFIG['min_seg_len']

    scores = jax.random.normal(k1, (num_rows, total_cols), dtype=dtype)

    raw_weights = jax.random.uniform(k2, (num_segments,), dtype=jnp.float32) + 0.1
    rem_cols = total_cols - num_segments * min_seg_len
    extra_lens = jnp.floor(raw_weights / jnp.sum(raw_weights) * rem_cols).astype(jnp.int32)
    seg_lens = min_seg_len + extra_lens
    diff = total_cols - jnp.sum(seg_lens)
    seg_lens = seg_lens.at[-1].add(diff)

    segment_offsets = jnp.concatenate([
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.cumsum(seg_lens, dtype=jnp.int32),
    ])
    return scores, segment_offsets


def workload(scores, segment_offsets):
    num_segments = CONFIG['num_segments']
    total_cols = scores.shape[-1]
    scale = CONFIG['scale']

    col_indices = jnp.arange(total_cols, dtype=jnp.int32)
    segment_ids = jnp.searchsorted(segment_offsets[1:-1], col_indices, side='right')

    scores_f32 = scores.astype(jnp.float32) * scale

    def row_segmented_softmax(row):
        seg_max = jax.ops.segment_max(row, segment_ids, num_segments=num_segments)
        shifted = row - seg_max[segment_ids]
        exp_scores = jnp.exp(shifted)
        seg_sum = jax.ops.segment_sum(exp_scores, segment_ids, num_segments=num_segments)
        return exp_scores / seg_sum[segment_ids]

    out = jax.vmap(row_segmented_softmax)(scores_f32)
    return out.astype(scores.dtype)
