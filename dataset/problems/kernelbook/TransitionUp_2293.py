"""TransitionUp: bilinear align_corners upsample of x to skip's spatial size, then channel concat with skip."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TransitionUp_2293',
    'batch': 8,
    'x_channels': 128,
    'x_height': 128,
    'x_width': 128,
    'skip_channels': 128,
    'skip_height': 256,
    'skip_width': 256,
    'concat': True,
    'channel_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    key_x, key_skip = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        key_x,
        (CONFIG['batch'], CONFIG['x_channels'], CONFIG['x_height'], CONFIG['x_width']),
        dtype,
    )
    skip = jax.random.uniform(
        key_skip,
        (CONFIG['batch'], CONFIG['skip_channels'], CONFIG['skip_height'], CONFIG['skip_width']),
        dtype,
    )
    return x, skip


def _linear_indices(in_size, out_size, dtype):
    if out_size > 1:
        coords = jnp.arange(out_size, dtype=dtype) * (in_size - 1) / (out_size - 1)
    else:
        coords = jnp.zeros((out_size,), dtype=dtype)
    low = jnp.clip(jnp.floor(coords).astype(jnp.int32), 0, in_size - 1)
    high = jnp.minimum(low + 1, in_size - 1)
    weight = coords - low.astype(dtype)
    return low, high, weight


def workload(x, skip):
    height_axis, width_axis = x.ndim - 2, x.ndim - 1
    out_height, out_width = skip.shape[height_axis], skip.shape[width_axis]

    low, high, weight = _linear_indices(x.shape[height_axis], out_height, x.dtype)
    weight = weight[:, None]
    resized = (
        jnp.take(x, low, axis=height_axis) * (1 - weight)
        + jnp.take(x, high, axis=height_axis) * weight
    )

    low, high, weight = _linear_indices(x.shape[width_axis], out_width, x.dtype)
    out = (
        jnp.take(resized, low, axis=width_axis) * (1 - weight)
        + jnp.take(resized, high, axis=width_axis) * weight
    )

    if CONFIG['concat']:
        out = jnp.concatenate([out, skip], axis=CONFIG['channel_axis'])
    return out
