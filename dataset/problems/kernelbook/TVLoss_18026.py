"""[bf16 variant] TVLoss: L1 norm of differences between abs finite differences of y and yhat along H and W."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TVLoss_18026',
    'batch': 8,
    'channels': 16,
    'height': 512,
    'width': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    yhat = jax.random.uniform(k1, shape, dtype)
    y = jax.random.uniform(k2, shape, dtype)
    return yhat, y


def workload(yhat, y):
    height, width = y.shape[2], y.shape[3]
    dyh = jnp.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    dyhath = jnp.abs(yhat[:, :, 1:, :] - yhat[:, :, :-1, :])
    dyw = jnp.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
    dyhatw = jnp.abs(yhat[:, :, :, 1:] - yhat[:, :, :, :-1])
    return (jnp.sum(jnp.abs(dyh - dyhath)) / height
            + jnp.sum(jnp.abs(dyw - dyhatw)) / width)
