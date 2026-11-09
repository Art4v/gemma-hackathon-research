"""[bf16 variant] Global average pool over length (AdaptiveAvgPool1d(1)) -> flatten -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResnetDecoder_17904',
    'batch': 32,
    'in_features': 2048,
    'length': 1024,
    'n_classes': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['in_features'], CONFIG['length']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_features'], CONFIG['n_classes']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['n_classes'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    pooled = jnp.mean(x, axis=-1)
    return pooled @ w + b
