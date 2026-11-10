"""[bf16 variant] GeM pooling: clamp(min=eps) -> pow(p) -> global spatial average -> pow(1/p)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GeM_16928',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'p': 3,
    'eps': 1e-06,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    p = CONFIG['p']
    clamped = jnp.maximum(x, jnp.asarray(CONFIG['eps'], x.dtype))
    pooled = jnp.mean(jnp.power(clamped, p), axis=(-2, -1), keepdims=True)
    return jnp.power(pooled, 1.0 / p)
