"""TransitionUpB: bilinear upsample (half-pixel, align_corners=False) -> center crop to skip -> concat with skip on channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TransitionUpB_6082',
    'batch': 8,
    'in_channels': 128,
    'height': 256,
    'width': 256,
    'skip_channels': 128,
    'skip_height': 256,
    'skip_width': 256,
    'scale_factor': 2,
    'channel_axis': 1,
    'height_axis': 2,
    'width_axis': 3,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_skip = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    skip = jax.random.uniform(
        k_skip,
        (CONFIG['batch'], CONFIG['skip_channels'], CONFIG['skip_height'], CONFIG['skip_width']),
        dtype,
    )
    return x, skip


def _bilinear_axis(a, axis, scale):
    n = a.shape[axis]
    out_size = n * scale
    coords = (jnp.arange(out_size, dtype=a.dtype) + 0.5) / scale - 0.5
    coords = jnp.clip(coords, 0.0, n - 1)
    lo = jnp.floor(coords).astype(jnp.int32)
    hi = jnp.minimum(lo + 1, n - 1)
    weight_shape = [1] * a.ndim
    weight_shape[axis] = out_size
    w = (coords - lo.astype(a.dtype)).reshape(weight_shape)
    return jnp.take(a, lo, axis=axis) * (1.0 - w) + jnp.take(a, hi, axis=axis) * w


def workload(x, skip):
    scale = CONFIG['scale_factor']
    h_axis = CONFIG['height_axis']
    w_axis = CONFIG['width_axis']

    out = _bilinear_axis(x, h_axis, scale)
    out = _bilinear_axis(out, w_axis, scale)

    max_height = skip.shape[h_axis]
    max_width = skip.shape[w_axis]
    xy1 = (out.shape[w_axis] - max_width) // 2
    xy2 = (out.shape[h_axis] - max_height) // 2
    out = out[:, :, xy2:xy2 + max_height, xy1:xy1 + max_width]

    return jnp.concatenate([out, skip], axis=CONFIG['channel_axis'])
