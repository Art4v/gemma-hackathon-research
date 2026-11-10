"""BaseQuantization: divide by step, round with STE, clip to [-(q_p - 1) * is_signed, q_p], multiply by step."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BaseQuantization_14536',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'symmetric': False,
    'delta_min': 1e-3,
    'delta_max': 1.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_delta, k_q_p, k_is_signed = jax.random.split(key, 4)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k_x, shape, dtype)
    delta = jax.random.uniform(
        k_delta, shape, dtype, minval=CONFIG['delta_min'], maxval=CONFIG['delta_max']
    )
    q_p = jax.random.uniform(k_q_p, shape, dtype)
    is_signed = jax.random.uniform(k_is_signed, shape, dtype)
    return x, delta, q_p, is_signed


def workload(x, delta, q_p, is_signed):
    symmetric_q_p = (q_p - int(not CONFIG['symmetric'])) * is_signed
    scaled = x / delta
    rounded = scaled + jax.lax.stop_gradient(jnp.round(scaled) - scaled)
    clipped = jnp.minimum(rounded, q_p)
    clipped = jnp.maximum(clipped, -symmetric_q_p)
    return delta * clipped
