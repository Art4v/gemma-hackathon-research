"""[bf16 variant] distLinear: L2-normalize input over axis 1, weight-normalized linear projection, scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'distLinear_14902',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'indim': 512,
    'outdim': 256,
    'eps': 1e-05,
    'outdim_threshold': 200,
    'scale_small': 2,
    'scale_large': 10,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_v = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['indim']),
        dtype,
    )
    weight_v = jax.random.normal(k_v, (CONFIG['indim'], CONFIG['outdim']), dtype) * 0.02
    weight_g = jnp.ones((1, CONFIG['outdim']), dtype)
    return x, weight_v, weight_g


def workload(x, weight_v, weight_g):
    outdim = weight_v.shape[1]
    scale = CONFIG['scale_small'] if outdim <= CONFIG['outdim_threshold'] else CONFIG['scale_large']
    x_norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))
    x_normalized = x / (x_norm + CONFIG['eps'])
    v_norm = jnp.sqrt(jnp.sum(weight_v * weight_v, axis=0, keepdims=True))
    weight = weight_v * (weight_g / v_norm)
    cos_dist = x_normalized @ weight
    return scale * cos_dist
