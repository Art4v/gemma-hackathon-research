"""L2 normalize along axis 1, bias-free linear projection on the last axis, divide by temperature."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Prototypes_4126',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'fdim': 512,
    'num_classes': 256,
    'temp': 0.05,
    'eps': 1e-12,
    'normalize_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['fdim']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['fdim'], CONFIG['num_classes']), dtype) * 0.02
    return x, w


def workload(x, w):
    norm = jnp.sqrt(jnp.sum(x * x, axis=CONFIG['normalize_axis'], keepdims=True))
    x = x / jnp.maximum(norm, CONFIG['eps'])
    out = x @ w
    return out / CONFIG['temp']
