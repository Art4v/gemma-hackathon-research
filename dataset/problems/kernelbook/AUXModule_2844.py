"""[bf16 variant] Global adaptive max pool over spatial axes (NCHW), flatten to (N, C), then linear layer with bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AUXModule_2844',
    'batch': 8,
    'in_features': 128,
    'height': 256,
    'width': 256,
    'out_features': 128,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_features'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['in_features'], CONFIG['out_features']), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (CONFIG['out_features'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    pooled = jnp.max(x, axis=(2, 3))
    flat = jnp.reshape(pooled, (-1, x.shape[1]))
    return flat @ w + b
