"""Ragged segment sum and segment mean over a packed token tensor given sequence lengths."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ragged_sequence_segment_sum_mean',
    'total_tokens': 65536,
    'hidden_dim': 4096,
    'num_segments': 256,
    'min_seg_len': 64,
    'max_seg_len': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    k_tokens, k_lens = jax.random.split(jax.random.key(0), 2)
    total_tokens = CONFIG['total_tokens']
    hidden_dim = CONFIG['hidden_dim']
    num_segments = CONFIG['num_segments']

    tokens = jax.random.normal(k_tokens, (total_tokens, hidden_dim), dtype=dtype)

    raw_lens = jax.random.randint(
        k_lens,
        (num_segments,),
        CONFIG['min_seg_len'],
        CONFIG['max_seg_len'] + 1,
        dtype=jnp.int32,
    )
    scaled_lens = jnp.floor(
        raw_lens.astype(jnp.float32) * (total_tokens / jnp.sum(raw_lens).astype(jnp.float32))
    ).astype(jnp.int32)
    scaled_lens = jnp.maximum(scaled_lens, 1)
    remainder = total_tokens - jnp.sum(scaled_lens[:-1])
    seqlens = scaled_lens.at[-1].set(remainder)

    return tokens, seqlens


def workload(tokens, seqlens):
    num_segments = CONFIG['num_segments']
    total_tokens = CONFIG['total_tokens']

    cu_seqlens = jnp.cumsum(seqlens)
    token_indices = jnp.arange(total_tokens, dtype=jnp.int32)
    segment_ids = jnp.searchsorted(cu_seqlens, token_indices, side='right')

    tokens_f32 = tokens.astype(jnp.float32)
    seg_sum_f32 = jax.ops.segment_sum(
        tokens_f32,
        segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
    )
    inv_lens = 1.0 / jnp.maximum(seqlens.astype(jnp.float32), 1.0)[:, None]
    seg_mean_f32 = seg_sum_f32 * inv_lens

    return seg_sum_f32.astype(tokens.dtype), seg_mean_f32.astype(tokens.dtype)
