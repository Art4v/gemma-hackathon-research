"""FilterResponseNorm: x / sqrt(mean_hw(x^2) + eps), then TLU relu(scale*x + bias - tau) + tau."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FilterResponseNorm_3200',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-16,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    c = CONFIG['channels']
    scale = jnp.ones((c,), dtype)
    bias = jnp.zeros((c,), dtype)
    threshold = jnp.zeros((c,), dtype)
    return x, scale, bias, threshold


def workload(x, scale, bias, threshold):
    spatial_axes = tuple(range(2, x.ndim))
    nu2 = jnp.mean(x * x, axis=spatial_axes, keepdims=True)
    out = x / jnp.sqrt(nu2 + CONFIG['eps'])
    bshape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    s = scale.reshape(bshape)
    b = bias.reshape(bshape)
    t = threshold.reshape(bshape)
    return jax.nn.relu(s * out + b - t) + t
