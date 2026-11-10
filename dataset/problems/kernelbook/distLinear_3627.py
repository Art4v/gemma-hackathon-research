"""L2-normalize over dim 1 -> weight-normed linear (v * g / ||v||) on last axis -> scale (distLinear cosine classifier)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'distLinear_3627',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'indim': 256,
    'outdim': 128,
    'eps': 1e-05,
    'weight_std': 0.02,
    'outdim_threshold': 200,
    'scale_small': 2,
    'scale_large': 10,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_v = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['indim']),
        dtype,
    )
    weight_v = jax.random.normal(k_v, (CONFIG['indim'], CONFIG['outdim']), dtype) * CONFIG['weight_std']
    weight_g = jnp.ones((1, CONFIG['outdim']), dtype)
    return x, weight_v, weight_g


def workload(x, weight_v, weight_g):
    x_norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))
    x_normalized = x / (x_norm + CONFIG['eps'])
    v_norm = jnp.sqrt(jnp.sum(weight_v * weight_v, axis=0, keepdims=True))
    weight = weight_v * (weight_g / v_norm)
    cos_dist = jnp.matmul(x_normalized, weight)
    outdim = weight_v.shape[1]
    scale_factor = CONFIG['scale_small'] if outdim <= CONFIG['outdim_threshold'] else CONFIG['scale_large']
    return scale_factor * cos_dist
