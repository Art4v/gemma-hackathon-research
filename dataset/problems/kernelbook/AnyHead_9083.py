"""[bf16 variant] AnyHead: AdaptiveAvgPool2d((1,1)) -> flatten -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AnyHead_9083',
    'batch': 64,
    'w_in': 1024,
    'height': 32,
    'width': 32,
    'nc': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['w_in'], CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(k_w, (CONFIG['w_in'], CONFIG['nc']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['nc'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    pooled = jnp.mean(x, axis=(2, 3))
    return pooled @ w + b
