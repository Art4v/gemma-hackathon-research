"""[bf16 variant] AdaptiveAvgMaxPool2d (pool_type='avg', output_size=1): global average pooling over H, W."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AdaptiveAvgMaxPool2d_13347',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    'output_size': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    # AdaptiveAvgPool2d(1) on NCHW is a mean over the spatial axes, kept as 1x1.
    return jnp.mean(x, axis=(2, 3), keepdims=True)
