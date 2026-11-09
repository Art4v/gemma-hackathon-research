"""[bf16 variant] GeM pooling: clamp(min=eps) -> pow(p) -> global avg pool over HxW -> pow(1/p)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GeM_3478',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
    'p': 3,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    p = CONFIG['p']
    pooled = jnp.mean(jnp.maximum(x, CONFIG['eps']) ** p, axis=(2, 3), keepdims=True)
    return pooled ** (1.0 / p)
