"""[bf16 variant] SELayer: global average pool -> linear -> relu -> linear -> sigmoid -> channel-wise gating of NCHW input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SELayer_257',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'reduction': 4,
    'mid_channels': 32,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    channels = CONFIG['channels']
    mid_channels = CONFIG['mid_channels']
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], channels, CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w1 = jax.random.normal(k_w1, (channels, mid_channels), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (mid_channels,), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (mid_channels, channels), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (channels,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    y = jnp.mean(x, axis=(2, 3))
    y = jax.nn.relu(y @ w1 + b1)
    y = jax.nn.sigmoid(y @ w2 + b2)
    return x * y[:, :, None, None]
