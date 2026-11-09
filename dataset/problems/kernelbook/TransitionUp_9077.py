"""[bf16 variant] TransitionUp: bilinear upsample (align_corners=True) to skip's spatial size, then channel concat."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TransitionUp_9077',
    'batch': 8,
    'in_channels': 128,
    'skip_channels': 128,
    'in_h': 128,
    'in_w': 128,
    'out_h': 256,
    'out_w': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k1, (CONFIG['batch'], CONFIG['in_channels'], CONFIG['in_h'], CONFIG['in_w']), dtype)
    skip = jax.random.uniform(
        k2, (CONFIG['batch'], CONFIG['skip_channels'], CONFIG['out_h'], CONFIG['out_w']), dtype)
    return x, skip


def _interp_axis(x, out_size, axis):
    in_size = x.shape[axis]
    scale = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    pos = jnp.arange(out_size, dtype=jnp.float32) * jnp.float32(scale)
    i0 = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0, in_size - 1)
    i1 = jnp.minimum(i0 + 1, in_size - 1)
    w_shape = [1] * x.ndim
    w_shape[axis] = out_size
    w = (pos - i0.astype(jnp.float32)).reshape(w_shape).astype(x.dtype)
    a = jnp.take(x, i0, axis=axis)
    b = jnp.take(x, i1, axis=axis)
    return a + (b - a) * w


def workload(x, skip):
    out = _interp_axis(x, skip.shape[2], axis=2)
    out = _interp_axis(out, skip.shape[3], axis=3)
    return jnp.concatenate([out, skip], axis=1)
