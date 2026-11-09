"""[bf16 variant] AdaptiveMaxPool2d((1,1)) -> flatten -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AUXModule_17535',
    'batch': 128,
    'in_features': 512,
    'height': 32,
    'width': 32,
    'out_features': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['in_features'], CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_features'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    pooled = jnp.max(x, axis=(2, 3))  # adaptive_max_pool2d to (1,1) + view(-1, C)
    return pooled @ w + b
