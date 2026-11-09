"""TVLoss: sum(|x[h+1]-x[h]|) + sum(|x[w+1]-x[w]|) over an NCHW tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TVLoss_4504',
    'batch': 16,
    'channels': 16,
    'height': 512,
    'width': 512,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    h_tv = jnp.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    w_tv = jnp.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    return jnp.sum(h_tv) + jnp.sum(w_tv)
