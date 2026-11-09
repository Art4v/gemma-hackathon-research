"""[m1 M1_batch variant of authored/ragged_segment_reduce/packed_segment_sum_mean] Packed ragged segment sum and mean, vmapped over a leading batch of 4 independent packed rows, each with its own per-sequence lengths."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'packed_segment_sum_mean_m1_batch',
    'num_rows': 4,
    'total_tokens': 8192,
    'hidden_dim': 4096,
    'num_sequences': 64,
    'min_len_weight': 1,
    'max_len_weight': 64,
}


def create_inputs(dtype=jnp.float32):
    R = CONFIG['num_rows']
    T = CONFIG['total_tokens']
    D = CONFIG['hidden_dim']
    B = CONFIG['num_sequences']
    key = jax.random.key(0)
    k_x, k_len = jax.random.split(key, 2)
    tokens = jax.random.normal(k_x, (R, T, D), dtype=dtype)
    # Deterministic ragged lengths that sum exactly to T within every packed row.
    w = jax.random.randint(k_len, (R, B), CONFIG['min_len_weight'],
                           CONFIG['max_len_weight'] + 1, dtype=jnp.int32)
    lengths = (w.astype(jnp.int64 if jax.config.jax_enable_x64 else jnp.int32) * T) // jnp.sum(
        w, axis=-1, keepdims=True)
    lengths = lengths.astype(jnp.int32)
    remainder = T - jnp.sum(lengths, axis=-1)
    lengths = lengths.at[:, B - 1].add(remainder)
    return tokens, lengths


def _segment_sum_mean(tokens, lengths):
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


def workload(tokens, lengths):
    # tokens (R, T, D), lengths (R, B) -> sums and means (R, B, D).
    return jax.vmap(_segment_sum_mean)(tokens, lengths)
