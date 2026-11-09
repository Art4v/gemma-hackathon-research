"""Smooth loss: mean |dx| along W plus mean |dy| along H."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Smooth_Loss_17740',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 512,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    loss_w = jnp.mean(jnp.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    loss_h = jnp.mean(jnp.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return loss_w + loss_h
