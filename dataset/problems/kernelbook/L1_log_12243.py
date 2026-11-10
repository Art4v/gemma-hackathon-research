"""L1 log loss: optional bilinear resize of fake to real's size, then mean(abs(log(real) - log(fake)))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L1_log_12243',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'minval': 1e-3,
    'maxval': 1.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_fake, k_real = jax.random.split(key)
    fake = jax.random.uniform(k_fake, shape, dtype, minval=CONFIG['minval'], maxval=CONFIG['maxval'])
    real = jax.random.uniform(k_real, shape, dtype, minval=CONFIG['minval'], maxval=CONFIG['maxval'])
    return fake, real


def workload(fake, real):
    if fake.shape != real.shape:
        height, width = real.shape[2:]
        fake = jax.image.resize(
            fake,
            fake.shape[:2] + (height, width),
            method='linear',
            antialias=False,
        )
    return jnp.mean(jnp.abs(jnp.log(real) - jnp.log(fake)))
