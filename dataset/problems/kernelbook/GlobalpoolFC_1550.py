"""Global average pool (AdaptiveAvgPool2d(1)) -> flatten -> Linear, returning (pooled, logits)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GlobalpoolFC_1550',
    'batch': 32,
    'num_in': 512,
    'height': 64,
    'width': 64,
    'num_class': 1024,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['num_in'], CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(k_w, (CONFIG['num_in'], CONFIG['num_class']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['num_class'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    rep = jnp.mean(x, axis=(2, 3), keepdims=True)
    y = rep.reshape(rep.shape[0], -1) @ w + b
    return rep, y
