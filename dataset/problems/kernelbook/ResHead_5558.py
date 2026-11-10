"""ResHead: global average pool over spatial axes, flatten, affine linear layer."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResHead_5558',
    'batch': 16,
    'w_in': 256,
    'height': 128,
    'width': 128,
    'num_classes': 1024,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['w_in'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['w_in'], CONFIG['num_classes']), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (CONFIG['num_classes'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    pooled = jnp.mean(x, axis=(2, 3), keepdims=True)
    flat = jnp.reshape(pooled, (pooled.shape[0], -1))
    return flat @ w + b
