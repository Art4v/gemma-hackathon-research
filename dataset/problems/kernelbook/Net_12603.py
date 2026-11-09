"""[bf16 variant] MaxPool2d(2,2) -> view(-1, 768) -> Linear(768,64) -> ReLU -> Linear(64,10)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Net_12603',
    'batch': 8192,
    'channels': 4,
    'height': 24,
    'width': 24,
    'pool_kernel': 2,
    'pool_stride': 2,
    'in_features': 3 * 16 * 16,
    'hidden': 64,
    'out_features': 10,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    c = CONFIG
    x = jax.random.uniform(k_x, (c['batch'], c['channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k_w1, (c['in_features'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (c['hidden'], c['out_features']), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (c['out_features'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    k = CONFIG['pool_kernel']
    s = CONFIG['pool_stride']
    pooled = lax.reduce_window(
        x, -jnp.inf, lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding='VALID',
    )
    # Matches torch view(-1, 768): flattens the whole tensor, not per sample.
    flat = pooled.reshape(-1, CONFIG['in_features'])
    h = jax.nn.relu(flat @ w1 + b1)
    return h @ w2 + b2
