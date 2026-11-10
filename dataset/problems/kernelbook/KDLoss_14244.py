"""KDLoss: log_softmax(input / T, dim=1) -> KLDivLoss(sum) vs target -> * T^2 / batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KDLoss_14244',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'temp_factor': 4.0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return x, target


def workload(x, target):
    t = CONFIG['temp_factor']
    log_p = jax.nn.log_softmax(x / t, axis=1)
    # xlogy convention (PyTorch KLDivLoss): target == 0 contributes 0.
    pos = target > 0
    safe_t = jnp.where(pos, target, 1.0)
    pointwise = jnp.where(pos, target * (jnp.log(safe_t) - log_p), 0.0)
    return jnp.sum(pointwise) * (t ** 2) / x.shape[0]
