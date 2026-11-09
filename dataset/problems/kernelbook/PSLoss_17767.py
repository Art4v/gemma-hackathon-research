"""[bf16 variant] PSLoss: fft2 over spatial dims -> abs -> L1 (mean) loss between power spectra."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PSLoss_17767',
    'batch': 16,
    'channels': 16,
    'height': 256,
    'width': 256,
    'fft_axes': (2, 3),
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ky = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    axes = CONFIG['fft_axes']
    x_power = jnp.abs(jnp.fft.fftn(x, axes=axes))
    y_power = jnp.abs(jnp.fft.fftn(y, axes=axes))
    # L1Loss default reduction='mean'; the trailing .sum() on a scalar is a no-op.
    return jnp.mean(jnp.abs(x_power - y_power))
