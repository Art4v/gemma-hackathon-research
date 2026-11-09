"""AffineChannel2d: per-channel scale and shift, x * w[c] + b[c] on NCHW input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AffineChannel2d_11227',
    'batch': 16,
    'channels': 128,
    'height': 256,
    'width': 256,
    'eps': 1e-5,  # stored by the module, unused in forward
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype
    )
    # Matches torch init: weight.uniform_(), bias.zero_().
    weight = jax.random.uniform(k_w, (CONFIG['channels'],), dtype)
    bias = jnp.zeros((CONFIG['channels'],), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    c = x.shape[1]
    return x * weight.reshape(1, c, 1, 1) + bias.reshape(1, c, 1, 1)
