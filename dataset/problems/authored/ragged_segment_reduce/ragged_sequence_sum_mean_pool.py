"""Ragged segment sum and segment mean pooling over packed variable-length sequences."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ragged_sequence_sum_mean_pool',
    'total_tokens': 65536,
    'num_seqs': 256,
    'hidden_dim': 4096,
    'min_seq_len': 64,
    'max_seq_len': 448,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    k_act, k_len = jax.random.split(jax.random.key(0), 2)
    total_tokens = CONFIG['total_tokens']
    num_seqs = CONFIG['num_seqs']
    hidden_dim = CONFIG['hidden_dim']

    tokens = jax.random.normal(k_act, (total_tokens, hidden_dim), dtype=dtype)
    seq_lens = jax.random.randint(
        k_len,
        (num_seqs,),
        minval=CONFIG['min_seq_len'],
        maxval=CONFIG['max_seq_len'],
        dtype=jnp.int32,
    )
    return tokens, seq_lens


def workload(tokens, seq_lens):
    total_tokens = tokens.shape[0]
    num_seqs = seq_lens.shape[0]

    cu_seqlens = jnp.cumsum(seq_lens, dtype=jnp.int32)
    total_valid = cu_seqlens[-1]

    token_idx = jnp.arange(total_tokens, dtype=jnp.int32)
    segment_ids = jnp.searchsorted(cu_seqlens, token_idx, side='right')
    segment_ids = jnp.where(token_idx < total_valid, segment_ids, num_seqs)

    tokens_f32 = tokens.astype(jnp.float32)
    seg_sum = jax.ops.segment_sum(
        tokens_f32,
        segment_ids,
        num_segments=num_seqs,
        indices_are_sorted=True,
    )

    denom = jnp.maximum(seq_lens.astype(jnp.float32), CONFIG['eps'])[:, None]
    seg_mean = seg_sum / denom

    return seg_sum.astype(tokens.dtype), seg_mean.astype(tokens.dtype)
