"""[bf16 variant] AffineChannelwise: channelwise scale and shift on axis 1 (x * weight + bias)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AffineChannelwise_6585',
    'batch': 8,
    'num_channels': 128,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['num_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jnp.ones((CONFIG['num_channels'],), dtype)
    bias = jnp.zeros((CONFIG['num_channels'],), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    param_shape = [1] * x.ndim
    param_shape[1] = weight.shape[0]
    return x * weight.reshape(param_shape) + bias.reshape(param_shape)
