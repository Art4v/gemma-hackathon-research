"""[bf16 variant] Mask_1669: expand mask to (B, S, L) via unsqueeze, repeat and transpose, then where(mask == 1, seq, 0)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Mask_1669',
    'batch': 32,
    'seq_dim': 1024,
    'mask_len': 1024,
    'mask_high': 2,
    'mask_true_value': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_seq, k_mask = jax.random.split(key)
    seq = jax.random.uniform(
        k_seq,
        (CONFIG['batch'], CONFIG['seq_dim'], CONFIG['mask_len']),
        dtype,
    )
    mask = jax.random.randint(
        k_mask,
        (CONFIG['batch'], CONFIG['mask_len']),
        0,
        CONFIG['mask_high'],
    ).astype(dtype)
    return seq, mask


def workload(seq, mask):
    seq_mask = jnp.expand_dims(mask, 2)
    seq_mask = jnp.tile(seq_mask, (1, 1, seq.shape[1]))
    seq_mask = jnp.swapaxes(seq_mask, 1, 2)
    return jnp.where(seq_mask == CONFIG['mask_true_value'], seq, jnp.zeros_like(seq))
