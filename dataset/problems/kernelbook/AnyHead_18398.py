"""[bf16 variant] AnyHead: adaptive average pool to (1, 1) -> flatten -> linear with bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AnyHead_18398',
    'batch': 8,
    'w_in': 128,
    'height': 256,
    'width': 256,
    'nc': 1024,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['w_in'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['w_in'], CONFIG['nc']), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (CONFIG['nc'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    pooled = jnp.mean(x, axis=(2, 3), keepdims=True)
    flat = jnp.reshape(pooled, (pooled.shape[0], -1))
    return flat @ w + b
