"""[bf16 variant] Collapse: global avg pool + global max pool over HW, mixed by per-channel sigmoid gate."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Collapse_18331',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    weight = jax.random.normal(kw, (CONFIG['channels'],), dtype) * 0.02
    return x, weight


def workload(x, weight):
    p_avg = jnp.mean(x, axis=(2, 3), keepdims=True)
    p_max = jnp.max(x, axis=(2, 3), keepdims=True)
    factor = jax.nn.sigmoid(weight)[None, :, None, None]
    return p_avg * factor + p_max * (1.0 - factor)
