"""FocalLoss: binary_cross_entropy_with_logits -> exp(-bce) -> alpha*(1-pt)^gamma*bce -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_7163',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'alpha': 0.5,
    'gamma': 2,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    logits = jax.random.uniform(k1, shape, dtype)
    y = jax.random.uniform(k2, shape, dtype)
    return logits, y


def workload(logits, y):
    # Same stable formulation PyTorch uses for BCE-with-logits.
    bce = jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    pt = jnp.exp(-bce)
    focal = CONFIG['alpha'] * (1 - pt) ** CONFIG['gamma'] * bce
    return jnp.mean(focal)
