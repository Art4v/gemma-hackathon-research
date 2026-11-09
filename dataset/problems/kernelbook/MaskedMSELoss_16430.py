"""[bf16 variant] Masked BCEWithLogits (sum reduction) divided by mask sum: loss(pred*mask, target) / sum(mask)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MaskedMSELoss_16430',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    pred = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    mask = jax.random.uniform(k3, shape, dtype)
    return pred, target, mask


def workload(pred, target, mask):
    x = pred * mask
    # Same numerically stable form PyTorch uses for BCEWithLogitsLoss.
    per_elem = jnp.maximum(x, 0) - x * target + jnp.log1p(jnp.exp(-jnp.abs(x)))
    return jnp.sum(per_elem) / jnp.sum(mask)
