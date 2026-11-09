"""[bf16 variant] TVLoss: 2 * (sum((x[:,:,1:,:]-x[:,:,:-1,:])^2) + sum((x[:,:,:,1:]-x[:,:,:,:-1])^2))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TVLoss_530',
    'batch': 16,
    'channels': 16,
    'height': 512,
    'width': 512,
    'scale': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    h_tv = jnp.sum(jnp.square(x[:, :, 1:, :] - x[:, :, :-1, :]))
    w_tv = jnp.sum(jnp.square(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return CONFIG['scale'] * (h_tv + w_tv)
