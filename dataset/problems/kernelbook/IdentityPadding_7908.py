"""IdentityPadding: zero-pad the channel axis of an NCHW tensor (zero-width at the configured sizes, stride-1 path with no pooling)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'IdentityPadding_7908',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    add_channels = CONFIG['out_channels'] - CONFIG['in_channels']
    padding_config = [(0, 0, 0), (0, add_channels, 0), (0, 0, 0), (0, 0, 0)]
    return lax.pad(x, jnp.zeros((), x.dtype), padding_config)
