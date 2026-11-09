"""[bf16 variant] SELayer: global avg pool -> Linear -> ReLU -> Linear -> sigmoid -> channel-wise scale of x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SELayer_2477',
    'batch': 16,
    'in_channels': 512,
    'reduction': 4,
    'height': 64,
    'width': 64,
}


def create_inputs(dtype=jnp.bfloat16):
    n, c = CONFIG['batch'], CONFIG['in_channels']
    h, w = CONFIG['height'], CONFIG['width']
    mid = c // CONFIG['reduction']
    k = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k[0], (n, c, h, w), dtype)
    fc1_w = jax.random.normal(k[1], (c, mid), dtype) * 0.02
    fc1_b = jax.random.normal(k[2], (mid,), dtype) * 0.02
    fc2_w = jax.random.normal(k[3], (mid, c), dtype) * 0.02
    fc2_b = jax.random.normal(k[4], (c,), dtype) * 0.02
    return x, fc1_w, fc1_b, fc2_w, fc2_b


def workload(x, fc1_w, fc1_b, fc2_w, fc2_b):
    y = jnp.mean(x, axis=(2, 3))
    y = jax.nn.relu(y @ fc1_w + fc1_b)
    y = jax.nn.sigmoid(y @ fc2_w + fc2_b)
    return x * y[:, :, None, None]
