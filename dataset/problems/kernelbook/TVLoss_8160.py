"""[bf16 variant] Total variation loss: squared forward differences along H and W of an NCHW tensor, summed."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TVLoss_8160',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
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
    h_x = x.shape[2]
    w_x = x.shape[3]
    h_tv = jnp.sum(jnp.square(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
    w_tv = jnp.sum(jnp.square(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
    return h_tv + w_tv
