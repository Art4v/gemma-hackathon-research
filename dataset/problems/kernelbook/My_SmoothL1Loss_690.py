"""[bf16 variant] SmoothL1-style loss: sub -> abs threshold masks -> masked 0.5*z^2 mean + masked 0.01*(|z|-0.005) mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'My_SmoothL1Loss_690',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'threshold': 0.01,
    'l1_scale': 0.01,
    'l1_offset': 0.005,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    kx, ky = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    thr = CONFIG['threshold']
    z = (x - y).astype(jnp.float32)
    az = jnp.abs(z)
    mse_mask = (az < thr).astype(jnp.float32)
    l1_mask = (az >= thr).astype(jnp.float32)
    mse = mse_mask * z
    l1 = l1_mask * z
    mse_term = jnp.mean(0.5 * jnp.square(mse) * mse_mask)
    l1_term = jnp.mean(CONFIG['l1_scale'] * (jnp.abs(l1) - CONFIG['l1_offset']) * l1_mask)
    return mse_term + l1_term
