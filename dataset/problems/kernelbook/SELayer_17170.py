"""SELayer: adaptive_avg_pool2d -> Linear -> ReLU -> Linear -> sigmoid -> channel-wise scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SELayer_17170',
    'batch': 32,
    'in_channels': 1024,
    'height': 32,
    'width': 32,
    'reduction': 8,
}


def create_inputs(dtype=jnp.float32):
    b, c = CONFIG['batch'], CONFIG['in_channels']
    h, w = CONFIG['height'], CONFIG['width']
    mid = c // CONFIG['reduction']
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (b, c, h, w), dtype)
    w1 = jax.random.normal(k1, (c, mid), dtype) * 0.02
    b1 = jax.random.normal(kb1, (mid,), dtype) * 0.02
    w2 = jax.random.normal(k2, (mid, c), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    y = jnp.mean(x, axis=(2, 3))
    y = jax.nn.relu(y @ w1 + b1)
    y = jax.nn.sigmoid(y @ w2 + b2)
    return x * y[:, :, None, None]
