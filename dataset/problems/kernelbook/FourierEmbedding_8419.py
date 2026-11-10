"""[bf16 variant] FourierEmbedding: coordinate normalization -> stack (x, y) -> linear projection with bias -> sin."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FourierEmbedding_8419',
    'batch': 1024,
    'num_points': 6656,
    'coord_dims': 2,
    'features': 128,
    'height': 4,
    'width': 4,
    'norm_scale': 2.0,
    'norm_shift': 1.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_y, k_x, k_w, k_b = jax.random.split(key, 4)
    shape = (CONFIG['batch'], CONFIG['num_points'])
    y = jax.random.uniform(k_y, shape, dtype)
    x = jax.random.uniform(k_x, shape, dtype)
    weight = jax.random.normal(k_w, (CONFIG['coord_dims'], CONFIG['features']), dtype) * CONFIG['weight_scale']
    bias = jax.random.normal(k_b, (CONFIG['features'],), dtype) * CONFIG['weight_scale']
    return y, x, weight, bias


def workload(y, x, weight, bias):
    x_norm = CONFIG['norm_scale'] * x / (CONFIG['width'] - 1) - CONFIG['norm_shift']
    y_norm = CONFIG['norm_scale'] * y / (CONFIG['height'] - 1) - CONFIG['norm_shift']
    z = jnp.stack((x_norm, y_norm), axis=2)
    return jnp.sin(z @ weight + bias)
