"""AddCoords: build normalized x/y coordinate grids, broadcast to batch, concatenate onto NCHW channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AddCoords_9362',
    'batch': 8,
    'channels': 128,
    'x_dim': 256,
    'y_dim': 128,
    'with_r': False,
    'coord_scale': 2.0,
    'coord_shift': 1.0,
    'r_center': 0.5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['x_dim'], CONFIG['y_dim'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(input_tensor):
    batch_size, _, x_dim, y_dim = input_tensor.shape
    dtype = input_tensor.dtype
    scale = CONFIG['coord_scale']
    shift = CONFIG['coord_shift']

    xx = jnp.arange(x_dim, dtype=dtype) / (x_dim - 1) * scale - shift
    yy = jnp.arange(y_dim, dtype=dtype) / (y_dim - 1) * scale - shift

    coord_shape = (batch_size, 1, x_dim, y_dim)
    xx_channel = jnp.broadcast_to(xx[None, None, :, None], coord_shape)
    yy_channel = jnp.broadcast_to(yy[None, None, None, :], coord_shape)

    ret = jnp.concatenate([input_tensor, xx_channel, yy_channel], axis=1)

    if CONFIG['with_r']:
        center = CONFIG['r_center']
        rr = jnp.sqrt(jnp.square(xx_channel - center) + jnp.square(yy_channel - center))
        ret = jnp.concatenate([ret, rr], axis=1)

    return ret
