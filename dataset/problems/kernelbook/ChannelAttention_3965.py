"""ChannelAttention: global avg pool -> Linear -> ReLU -> Linear -> sigmoid -> scale input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ChannelAttention_3965',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
    'reduction': 4,
}


def create_inputs(dtype=jnp.float32):
    b, c = CONFIG['batch'], CONFIG['channels']
    h, w = CONFIG['height'], CONFIG['width']
    hidden = c // CONFIG['reduction']
    kx, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (b, c, h, w), dtype)
    fc1_w = jax.random.normal(k1, (c, hidden), dtype) * 0.02
    fc1_b = jax.random.normal(k2, (hidden,), dtype) * 0.02
    fc2_w = jax.random.normal(k3, (hidden, c), dtype) * 0.02
    fc2_b = jax.random.normal(k4, (c,), dtype) * 0.02
    return x, fc1_w, fc1_b, fc2_w, fc2_b


def workload(x, fc1_w, fc1_b, fc2_w, fc2_b):
    # ponytail: avg_pool2d(kernel=W) equals a global mean because CONFIG keeps H == W
    pooled = jnp.mean(x, axis=(2, 3))  # (B, C)
    hidden = jax.nn.relu(pooled @ fc1_w + fc1_b)
    atten = jax.nn.sigmoid(hidden @ fc2_w + fc2_b)  # (B, C)
    return atten[:, :, None, None] * x
