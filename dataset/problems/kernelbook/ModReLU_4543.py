"""ModReLU_4543: complex magnitude + eps, per-channel bias shift, threshold compare, where-zero mask of the input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ModReLU_4543',
    'batch': 16,
    'channels': 512,
    'height': 2048,
    'complex_dim': 2,
    'eps': 1e-05,
    'threshold': 0.3,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_b = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['complex_dim']),
        dtype,
    )
    b = jax.random.normal(k_b, (CONFIG['channels'],), dtype) * CONFIG['weight_scale']
    return x, b


def workload(x, b):
    mag = jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)[..., None] + CONFIG['eps']
    brdcst_b = b.reshape((1, -1) + (1,) * (x.ndim - 2))
    return jnp.where(mag + brdcst_b < CONFIG['threshold'], jnp.zeros((), x.dtype), x)
