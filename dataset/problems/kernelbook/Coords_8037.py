"""Coords: concat(x, linspace-H coordinate map, linspace-W coordinate map) along channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Coords_8037',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    b, _, h, w = x.shape
    hm = jnp.linspace(0.0, 1.0, h, dtype=x.dtype).reshape(1, 1, h, 1)
    wm = jnp.linspace(0.0, 1.0, w, dtype=x.dtype).reshape(1, 1, 1, w)
    hm = jnp.broadcast_to(hm, (b, 1, h, w))
    wm = jnp.broadcast_to(wm, (b, 1, h, w))
    return jnp.concatenate((x, hm, wm), axis=1)
