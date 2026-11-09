"""[bf16 variant] ResHead: global average pool (NCHW) -> flatten -> linear with bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResHead_11605',
    'batch': 64,
    'w_in': 2048,
    'height': 16,
    'width': 16,
    'num_classes': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['w_in'], CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(kw, (CONFIG['w_in'], CONFIG['num_classes']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['num_classes'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    pooled = jnp.mean(x, axis=(2, 3))
    return pooled @ w + b
