"""[bf16 variant] RMSE of log: sqrt(mean((log(real) - log(fake))^2))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RMSE_log_717',
    'batch': 8,
    'channels': 16,
    'height': 512,
    'width': 512,
    'minval': 1e-3,  # keeps inputs off 0 so log stays finite
    'maxval': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_fake, k_real = jax.random.split(jax.random.key(0))
    fake = jax.random.uniform(k_fake, shape, dtype, CONFIG['minval'], CONFIG['maxval'])
    real = jax.random.uniform(k_real, shape, dtype, CONFIG['minval'], CONFIG['maxval'])
    return fake, real


def workload(fake, real):
    # ponytail: torch's bilinear-upsample branch is unreachable since inputs always share a shape
    assert fake.shape == real.shape, 'fake and real must have the same shape'
    diff = jnp.log(real) - jnp.log(fake)
    return jnp.sqrt(jnp.mean(diff * diff))
