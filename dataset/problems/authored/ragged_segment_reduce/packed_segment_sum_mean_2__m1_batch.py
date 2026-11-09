"""[m1 M1_batch variant of authored/ragged_segment_reduce/packed_segment_sum_mean_2] Packed ragged segment sum and mean, vmapped over a leading batch of 2 independent packed rows, each with its own per-sequence lengths."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'packed_segment_sum_mean_m1_batch',
    'num_rows': 2,
    'total_tokens': 16384,
    'hidden_dim': 4096,
    'num_sequences': 64,
}


def create_inputs(dtype=jnp.float32):
    R = CONFIG['num_rows']
    T = CONFIG['total_tokens']
    D = CONFIG['hidden_dim']
    B = CONFIG['num_sequences']
    key = jax.random.key(0)
    k_x, k_len = jax.random.split(key)
    x = jax.random.normal(k_x, (R, T, D), dtype=dtype)
    # Random cut points per row -> lengths summing exactly to T (zero-length segments allowed).
    cuts = jnp.sort(jax.random.randint(k_len, (R, B - 1), 0, T + 1, dtype=jnp.int32), axis=-1)
    bounds = jnp.concatenate([jnp.zeros((R, 1), jnp.int32), cuts,
                              jnp.full((R, 1), T, jnp.int32)], axis=-1)
    lengths = jnp.diff(bounds, axis=-1).astype(jnp.int32)
    return x, lengths


def _segment_sum_mean(x, lengths):
    T = x.shape[0]
    B = lengths.shape[0]
    ends = jnp.cumsum(lengths.astype(jnp.int32))
    token_idx = jnp.arange(T, dtype=jnp.int32)
    seg_ids = jnp.searchsorted(ends, token_idx, side='right').astype(jnp.int32)
    sums = jax.ops.segment_sum(
        x.astype(jnp.float32), seg_ids, num_segments=B, indices_are_sorted=True
    )
    denom = jnp.maximum(lengths, 1).astype(jnp.float32)[:, None]
    means = sums / denom
    return sums.astype(x.dtype), means.astype(x.dtype)


def workload(x, lengths):
    # x (R, T, D), lengths (R, B) -> sums and means (R, B, D).
    return jax.vmap(_segment_sum_mean)(x, lengths)
