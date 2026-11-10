"""Mask_4270: expand_dims -> tile -> swapaxes -> equality -> broadcast where (seq masked to zeros)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Mask_4270',
    'batch': 8,
    'seq_len': 1,
    'mask_len': 2097152,
    'mask_low': 0,
    'mask_high': 2,
    'mask_on_value': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_seq, k_mask = jax.random.split(key)
    seq = jax.random.uniform(k_seq, (CONFIG['batch'], CONFIG['seq_len']), dtype)
    mask = jax.random.randint(
        k_mask,
        (CONFIG['batch'], CONFIG['mask_len']),
        CONFIG['mask_low'],
        CONFIG['mask_high'],
    ).astype(dtype)
    return seq, mask


def workload(seq, mask):
    seq_mask = jnp.expand_dims(mask, 2)
    seq_mask = jnp.swapaxes(jnp.tile(seq_mask, (1, 1, seq.shape[1])), 1, 2)
    return jnp.where(seq_mask == CONFIG['mask_on_value'], seq, jnp.zeros_like(seq))
