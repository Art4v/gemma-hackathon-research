"""[bf16 variant] TransitionUp: bilinear upsample (align_corners=True) of x to skip's HxW, then channel concat with skip."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TransitionUp_15856',
    'batch': 16,
    'x_channels': 256,
    'x_height': 64,
    'x_width': 64,
    'skip_channels': 128,
    'skip_height': 128,
    'skip_width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ks = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['x_channels'], CONFIG['x_height'], CONFIG['x_width']), dtype)
    skip = jax.random.uniform(
        ks, (CONFIG['batch'], CONFIG['skip_channels'], CONFIG['skip_height'], CONFIG['skip_width']), dtype)
    return (x, skip)


def _lerp_axis(a, n_out, axis):
    # align_corners=True: src = i * (in - 1) / (out - 1); scale is 0 when out == 1 (PyTorch rule).
    n_in = a.shape[axis]
    scale = (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
    src = jnp.arange(n_out, dtype=jnp.float32) * scale
    i0 = jnp.minimum(jnp.floor(src).astype(jnp.int32), n_in - 1)
    i1 = jnp.minimum(i0 + 1, n_in - 1)
    frac = (src - i0.astype(jnp.float32)).astype(a.dtype)
    shape = [1] * a.ndim
    shape[axis] = n_out
    frac = frac.reshape(shape)
    lo = jnp.take(a, i0, axis=axis)
    hi = jnp.take(a, i1, axis=axis)
    return lo * (1 - frac) + hi * frac


def workload(x, skip):
    out_h, out_w = skip.shape[2], skip.shape[3]
    out = _lerp_axis(x, out_h, axis=2)
    out = _lerp_axis(out, out_w, axis=3)
    return jnp.concatenate([out, skip], axis=1)
