"""[bf16 variant] Bilinear upsample (scale 2, align_corners=True); quant/dequant stubs are identity."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'UpsamplingBilinear_6462',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'scale_factor': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def _interp_axis(x, axis, out_size):
    # align_corners=True: src = dst * (in - 1) / (out - 1)
    in_size = x.shape[axis]
    scale = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    src = jnp.arange(out_size, dtype=jnp.float32) * scale
    i0 = jnp.minimum(jnp.floor(src).astype(jnp.int32), in_size - 1)
    i1 = jnp.minimum(i0 + 1, in_size - 1)
    shape = [1] * x.ndim
    shape[axis] = out_size
    w1 = (src - i0.astype(jnp.float32)).astype(x.dtype).reshape(shape)
    return jnp.take(x, i0, axis=axis) * (1 - w1) + jnp.take(x, i1, axis=axis) * w1


def workload(x):
    s = CONFIG['scale_factor']
    out_h = int(x.shape[2] * s)
    out_w = int(x.shape[3] * s)
    y = _interp_axis(x, 2, out_h)
    return _interp_axis(y, 3, out_w)
