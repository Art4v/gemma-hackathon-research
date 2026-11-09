"""SpatialShift2d: replicate pad -> split channels in 4 -> shift left/right/up/down -> concat."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SpatialShift2d_13065',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pad': 1,
    'padding_mode': 'edge',  # torch 'replicate'
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
    c = x.shape[1]
    qc = c // 4
    p = CONFIG['pad']
    xp = jnp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode=CONFIG['padding_mode'])
    l = xp[:, 0:qc, 1:-1, 0:-2]
    r = xp[:, qc:2 * qc, 1:-1, 2:]
    u = xp[:, 2 * qc:3 * qc, 0:-2, 1:-1]
    d = xp[:, 3 * qc:, 2:, 1:-1]
    return jnp.concatenate([l, r, u, d], axis=1)
