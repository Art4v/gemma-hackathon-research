"""Distribution_Loss: adaptive_avg_pool2d(1x1) -> softmax(dim=1) on both inputs -> MSE (mean)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Distribution_Loss_14357',
    'batch': 32,
    'channels': 512,
    'height': 64,
    'width': 64,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    input_logits = jax.random.uniform(k1, shape, dtype)
    target_logits = jax.random.uniform(k2, shape, dtype)
    return input_logits, target_logits


def workload(input_logits, target_logits):
    # adaptive_avg_pool2d to (1, 1) followed by reshape (N, -1) == mean over H, W.
    x = jnp.mean(input_logits, axis=(2, 3))
    y = jnp.mean(target_logits, axis=(2, 3))
    x_sm = jax.nn.softmax(x, axis=1)
    y_sm = jax.nn.softmax(y, axis=1)
    return jnp.mean((x_sm - y_sm) ** 2)
