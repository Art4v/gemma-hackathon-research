"""SoftExp activation: clamp alpha, compute exp/log branches, select per-element by the sign of alpha."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftExp_15743',
    'batch': 8,
    'height': 256,
    'width': 256,
    'input_size': 128,
    'alpha_min': -1.0,
    'alpha_max': 1.0,
    'threshold': 0.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    k_data, k_alpha = jax.random.split(jax.random.key(0))
    data = jax.random.uniform(
        k_data,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['input_size']),
        dtype,
    )
    alpha = jax.random.normal(k_alpha, (CONFIG['input_size'],), dtype) * CONFIG['weight_scale']
    return data, alpha


def workload(data, alpha):
    alpha = jnp.clip(alpha, CONFIG['alpha_min'], CONFIG['alpha_max'])
    pos_out = (jnp.exp(alpha * data) - 1) / alpha + alpha
    neg_out = -jnp.log(1 - alpha * (data + alpha)) / alpha
    output = jnp.where(alpha > CONFIG['threshold'], pos_out, data)
    output = jnp.where(alpha < -CONFIG['threshold'], neg_out, output)
    return output
