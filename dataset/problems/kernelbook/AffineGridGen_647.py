"""AffineGridGen: F.affine_grid(theta, (B, C, H, W)) with align_corners=False."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AffineGridGen_647',
    'batch': 512,
    'out_ch': 3,
    'out_h': 240,
    'out_w': 240,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    theta = jax.random.uniform(key, (CONFIG['batch'], 2, 3), dtype=dtype)
    return (theta,)


def _base_coords(n, dtype):
    # align_corners=False: pixel centers, linspace(-1, 1, n) * (n - 1) / n
    return jnp.linspace(-1.0, 1.0, n, dtype=dtype) * ((n - 1) / n)


def workload(theta):
    theta = theta.reshape(-1, 2, 3)
    h, w = CONFIG['out_h'], CONFIG['out_w']
    xs = _base_coords(w, theta.dtype)[None, None, :, None]
    ys = _base_coords(h, theta.dtype)[None, :, None, None]
    t0 = theta[:, None, None, :, 0]
    t1 = theta[:, None, None, :, 1]
    t2 = theta[:, None, None, :, 2]
    return xs * t0 + ys * t1 + t2
