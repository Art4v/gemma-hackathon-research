"""Ragged sequence segment sum and segment mean over a packed token tensor given sequence lengths."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ragged_sequence_segment_sum_mean',
    'total_tokens': 65536,
    'hidden_dim': 4096,
    'num_seqs': 256,
    'min_seq_len': 64,
    'max_seq_len': 256,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    total_tokens = CONFIG['total_tokens']
    hidden_dim = CONFIG['hidden_dim']
    num_seqs = CONFIG['num_seqs']
    min_seq_len = CONFIG['min_seq_len']
    max_seq_len = CONFIG['max_seq_len']

    tokens = jax.random.normal(k1, (total_tokens, hidden_dim), dtype=dtype)
    seq_lens = jax.random.randint(
        k2, (num_seqs,), minval=min_seq_len, maxval=max_seq_len + 1, dtype=jnp.int32
    )
    return tokens, seq_lens


def workload(tokens, seq_lens):
    total_tokens = tokens.shape[0]
    num_seqs = seq_lens.shape[0]

    cu_seqlens = jnp.cumsum(seq_lens, dtype=jnp.int32)
    token_indices = jnp.arange(total_tokens, dtype=jnp.int32)
    segment_ids = jnp.searchsorted(cu_seqlens, token_indices, side='right')

    tokens_f32 = tokens.astype(jnp.float32)
    seg_sum_f32 = jax.ops.segment_sum(
        tokens_f32,
        segment_ids,
        num_segments=num_seqs,
        indices_are_sorted=True,
    )
    denom = jnp.maximum(seq_lens[:, None].astype(jnp.float32), CONFIG['eps'])
    seg_mean_f32 = seg_sum_f32 / denom

    return seg_sum_f32.astype(tokens.dtype), seg_mean_f32.astype(tokens.dtype)
