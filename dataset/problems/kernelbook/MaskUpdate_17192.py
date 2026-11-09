"""[bf16 variant] MaskUpdate: pow(relu(x), clamp(alpha, 0.6, 0.8))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MaskUpdate_17192',
    'batch': 8,
    'channels': 32,
    'height': 512,
    'width': 512,
    'alpha': 4.0,
    'alpha_min': 0.6,
    'alpha_max': 0.8,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    alpha = jnp.asarray(CONFIG['alpha'], dtype=dtype)
    return x, alpha


def workload(x, alpha):
    a = jnp.clip(alpha, CONFIG['alpha_min'], CONFIG['alpha_max'])
    return jnp.power(jax.nn.relu(x), a)
