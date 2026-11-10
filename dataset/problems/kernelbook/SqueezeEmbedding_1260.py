"""[bf16 variant] SqueezeEmbedding: argsort by descending length -> gather -> truncate to longest length -> zero padded steps -> unsort."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SqueezeEmbedding_1260',
    'batch': 8,
    'seq_len': 512,
    'max_len': 384,
    'height': 128,
    'width': 128,
    'min_len': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_len = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['seq_len'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    x_len = jax.random.randint(
        k_len,
        (CONFIG['batch'],),
        CONFIG['min_len'],
        CONFIG['max_len'] + 1,
        dtype=jnp.int32,
    )
    x_len = x_len.at[0].set(CONFIG['max_len'])
    return x, x_len


def workload(x, x_len):
    max_len = min(CONFIG['max_len'], x.shape[1])
    sort_idx = jnp.argsort(-x_len)
    unsort_idx = jnp.argsort(sort_idx)
    sorted_len = x_len[sort_idx]
    sorted_x = x[sort_idx]
    truncated = sorted_x[:, :max_len]
    steps = jnp.arange(max_len, dtype=x_len.dtype)
    mask = steps[None, :] < sorted_len[:, None]
    mask = mask.reshape(mask.shape + (1,) * (x.ndim - 2))
    out = jnp.where(mask, truncated, jnp.zeros((), truncated.dtype))
    return out[unsort_idx]
