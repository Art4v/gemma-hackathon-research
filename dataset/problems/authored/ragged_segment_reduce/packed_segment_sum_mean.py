"""Packed ragged segment reduction: lengths -> segment ids -> per-sequence segment sum and segment mean over tokens."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'packed_segment_sum_mean',
    'total_tokens': 32768,
    'hidden_dim': 4096,
    'num_sequences': 64,
    'min_len_weight': 1,
    'max_len_weight': 64,
}


def create_inputs(dtype=jnp.float32):
    T = CONFIG['total_tokens']
    D = CONFIG['hidden_dim']
    B = CONFIG['num_sequences']
    key = jax.random.key(0)
    k_x, k_len = jax.random.split(key, 2)
    tokens = jax.random.normal(k_x, (T, D), dtype=dtype)
    # Deterministic ragged lengths that sum exactly to T.
    w = jax.random.randint(k_len, (B,), CONFIG['min_len_weight'],
                           CONFIG['max_len_weight'] + 1, dtype=jnp.int32)
    lengths = (w.astype(jnp.int64 if jax.config.jax_enable_x64 else jnp.int32) * T) // jnp.sum(w)
    lengths = lengths.astype(jnp.int32)
    remainder = T - jnp.sum(lengths)
    lengths = lengths.at[B - 1].add(remainder)
    return tokens, lengths


def workload(tokens, lengths):
    T = tokens.shape[0]
    B = lengths.shape[0]
    lengths = lengths.astype(jnp.int32)
    offsets = jnp.cumsum(lengths)  # exclusive end offset of each sequence
    positions = jnp.arange(T, dtype=jnp.int32)
    # Token t belongs to the first sequence whose end offset exceeds t; tokens past the end get id B (dropped).
    segment_ids = jnp.searchsorted(offsets, positions, side='right').astype(jnp.int32)
    seg_sum = jax.ops.segment_sum(tokens.astype(jnp.float32), segment_ids,
                                  num_segments=B, indices_are_sorted=True)
    counts = jnp.maximum(lengths, 1).astype(jnp.float32)[:, None]
    seg_mean = seg_sum / counts
    return seg_sum.astype(tokens.dtype), seg_mean.astype(tokens.dtype)
