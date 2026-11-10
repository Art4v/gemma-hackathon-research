"""[bf16 variant] Charbonnier loss: sum(sqrt((x - y)^2 + eps^2)) * out_norm (bci -> mean)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CharbonnierLoss_15458',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'eps': 1e-6,
    'out_norm': 'bci',
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    kx, ky = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def _outnorm(shape, out_norm):
    norm = 1.0
    if 'b' in out_norm:
        norm /= shape[0]
    if 'c' in out_norm:
        norm /= shape[-3]
    if 'i' in out_norm:
        norm /= shape[-1] * shape[-2]
    return norm


def workload(x, y):
    eps = CONFIG['eps']
    loss = jnp.sum(jnp.sqrt(jnp.square(x - y) + eps ** 2))
    return loss * _outnorm(x.shape, CONFIG['out_norm'])
