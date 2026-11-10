"""[bf16 variant] Focal loss: binary cross entropy with logits (mean) -> alpha * (1 - exp(-ce)) ** gamma * ce."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_9714',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'alpha': 1.0,
    'gamma': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_x, k_y = jax.random.split(key)
    x = jax.random.uniform(k_x, shape, dtype)
    y = jax.random.uniform(k_y, shape, dtype)
    return x, y


def workload(x, y):
    ce = jnp.mean((1 - y) * x - jax.nn.log_sigmoid(x))
    return CONFIG['alpha'] * (1 - jnp.exp(-ce)) ** CONFIG['gamma'] * ce
