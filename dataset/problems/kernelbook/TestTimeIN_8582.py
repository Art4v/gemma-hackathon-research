"""Test-time instance norm: per-channel spatial mean/variance standardize, affine scale/shift, upper clamp."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TestTimeIN_8582',
    'batch': 8,
    'num_features': 128,
    'height': 256,
    'width': 256,
    'eps': 1e-05,
    'clamp_max': 1.0,
    'spatial_axes': (2, 3),
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['num_features'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jnp.ones((CONFIG['num_features'],), dtype)
    bias = jnp.zeros((CONFIG['num_features'],), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    axes = CONFIG['spatial_axes']
    instance_var = jnp.var(x, axis=axes, keepdims=True)
    instance_mean = jnp.mean(x, axis=axes, keepdims=True)
    w = weight[None, :, None, None]
    b = bias[None, :, None, None]
    out = w * (x - instance_mean) / jnp.sqrt(instance_var + CONFIG['eps']) + b
    return jnp.minimum(out, CONFIG['clamp_max'])
