"""attention_11048: global AvgPool2d -> Linear -> ReLU -> Linear -> Sigmoid -> channel-wise multiply with NCHW input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'attention_11048',
    'batch': 8,
    'input_channels': 128,
    'hidden_channels': 64,
    'height': 256,
    'width': 256,
    'map_size': 256,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(key, 5)
    channels = CONFIG['input_channels']
    hidden = CONFIG['hidden_channels']
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], channels, CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w1 = jax.random.normal(k_w1, (channels, hidden), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (hidden,), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (hidden, channels), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (channels,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    pooled = jnp.mean(x, axis=(2, 3))
    h = jax.nn.relu(pooled @ w1 + b1)
    scale = jax.nn.sigmoid(h @ w2 + b2)
    scale = scale.reshape(scale.shape[0], scale.shape[1], 1, 1)
    return x * scale
