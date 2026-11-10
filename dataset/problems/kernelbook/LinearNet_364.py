"""[bf16 variant] LinearNet_364: L2 normalize (dim=1) -> Linear -> ReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearNet_364',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'd_in': 256,
    'hidden': 512,
    'd_out': 256,
    'norm_axis': 1,
    'eps': 1e-12,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    x_shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['d_in'])
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(k_x, x_shape, dtype)
    w1 = jax.random.normal(k_w1, (CONFIG['d_in'], CONFIG['hidden']), dtype) * scale
    b1 = jax.random.normal(k_b1, (CONFIG['hidden'],), dtype) * scale
    w2 = jax.random.normal(k_w2, (CONFIG['hidden'], CONFIG['d_out']), dtype) * scale
    b2 = jax.random.normal(k_b2, (CONFIG['d_out'],), dtype) * scale
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    norm = jnp.sqrt(jnp.sum(x * x, axis=CONFIG['norm_axis'], keepdims=True))
    x = x / jnp.maximum(norm, CONFIG['eps'])
    x = x @ w1 + b1
    x = jnp.maximum(x, 0)
    return x @ w2 + b2
