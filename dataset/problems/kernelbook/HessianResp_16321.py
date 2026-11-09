"""Hessian response: replicate-padded 3-tap derivative stencils, |gxx*gyy - gxy^2| * scale^4."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'HessianResp_16321',
    'batch': 8,
    'in_channels': 1,
    'scale_channels': 4,
    'height': 1024,
    'width': 1024,
    'gx': (0.5, 0.0, -0.5),
    'gy': (0.5, 0.0, -0.5),
    'gxx': (1.0, -2.0, 1.0),
    'gyy': (1.0, -2.0, 1.0),
    'scale_power': 4,
}


def create_inputs(dtype=jnp.float32):
    kx, ks = jax.random.split(jax.random.key(0))
    b, h, w = CONFIG['batch'], CONFIG['height'], CONFIG['width']
    x = jax.random.uniform(kx, (b, CONFIG['in_channels'], h, w), dtype)
    scale = jax.random.uniform(ks, (b, CONFIG['scale_channels'], h, w), dtype)
    gx = jnp.array(CONFIG['gx'], dtype)
    gy = jnp.array(CONFIG['gy'], dtype)
    gxx = jnp.array(CONFIG['gxx'], dtype)
    gyy = jnp.array(CONFIG['gyy'], dtype)
    return x, scale, gx, gy, gxx, gyy


def _stencil(x, k, axis):
    # Cross-correlation with a 3-tap kernel along `axis`, replicate (edge) padding of 1 on both sides.
    pad = [(0, 0)] * x.ndim
    pad[axis] = (1, 1)
    xp = jnp.pad(x, pad, mode='edge')
    n = x.shape[axis]
    taps = [lax_slice(xp, axis, i, n) for i in range(3)]
    return k[0] * taps[0] + k[1] * taps[1] + k[2] * taps[2]


def lax_slice(x, axis, start, size):
    return jax.lax.slice_in_dim(x, start, start + size, axis=axis)


def workload(x, scale, gx, gy, gxx, gyy):
    d_xx = _stencil(x, gxx, axis=3)
    d_yy = _stencil(x, gyy, axis=2)
    d_xy = _stencil(_stencil(x, gx, axis=3), gy, axis=2)
    return jnp.abs(d_xx * d_yy - d_xy * d_xy) * scale ** CONFIG['scale_power']
