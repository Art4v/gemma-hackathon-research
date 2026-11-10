"""[bf16 variant] Mask: expand_dims -> tile -> swapaxes -> equal -> where (broadcast seq over a repeated mask)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Mask_10990',
    'seq_size': 4096,
    'mask_batch': 8,
    'mask_low': 0,
    'mask_high': 2,
    'match_value': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_seq, k_mask = jax.random.split(key)
    n = CONFIG['seq_size']
    seq = jax.random.uniform(k_seq, (n, n), dtype)
    mask = jax.random.randint(
        k_mask, (CONFIG['mask_batch'], n), CONFIG['mask_low'], CONFIG['mask_high']
    ).astype(dtype)
    return seq, mask


def workload(seq, mask):
    seq_mask = jnp.expand_dims(mask, 2)
    seq_mask = jnp.swapaxes(jnp.tile(seq_mask, (1, 1, seq.shape[1])), 1, 2)
    return jnp.where(jnp.equal(seq_mask, CONFIG['match_value']), seq, jnp.zeros_like(seq))
