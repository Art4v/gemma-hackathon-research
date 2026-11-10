"""[bf16 variant] SimpleNet: NHWC-to-NCHW permute -> MaxPool2d -> flatten -> linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SimpleNet_9976',
    'batch': 8,
    'height': 256,
    'width': 256,
    'channels': 128,
    'pool_width': 4,
    'input_size': 524288,
    'output_size': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['channels']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['input_size'], CONFIG['output_size']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['output_size'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    p = CONFIG['pool_width']
    out = jnp.transpose(x, (0, 3, 1, 2))
    out = lax.reduce_window(out, -jnp.inf, lax.max, (1, 1, p, p), (1, 1, p, p), 'VALID')
    out = jnp.reshape(out, (out.shape[0], -1))
    return out @ w + b
