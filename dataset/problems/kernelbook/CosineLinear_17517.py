"""CosineLinear: L2-normalize input (axis 1) -> L2-normalize weight per class -> bias-free linear on last axis -> scale by sigma."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CosineLinear_17517',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'in_features': 256,
    'out_features': 256,
    'input_normalize_axis': 1,
    'weight_normalize_axis': 0,
    'eps': 1e-12,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['in_features']),
        dtype,
    )
    weight = jax.random.normal(
        k_w, (CONFIG['in_features'], CONFIG['out_features']), dtype
    ) * CONFIG['weight_scale']
    sigma = jnp.ones((1,), dtype)
    return x, weight, sigma


def workload(x, weight, sigma):
    eps = CONFIG['eps']
    x_axis = CONFIG['input_normalize_axis']
    w_axis = CONFIG['weight_normalize_axis']
    x_norm = jnp.sqrt(jnp.sum(x * x, axis=x_axis, keepdims=True))
    x_unit = x / jnp.maximum(x_norm, eps)
    w_norm = jnp.sqrt(jnp.sum(weight * weight, axis=w_axis, keepdims=True))
    w_unit = weight / jnp.maximum(w_norm, eps)
    return sigma * (x_unit @ w_unit)
