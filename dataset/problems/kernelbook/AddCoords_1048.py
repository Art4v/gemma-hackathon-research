"""[bf16 variant] AddCoords: build normalized x/y coordinate grids, optional radius channel, concatenate onto NCHW input along channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AddCoords_1048',
    'batch': 8,
    'channels': 128,
    'x_dim': 128,
    'y_dim': 256,
    'with_r': False,
    'r_center': 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['x_dim'], CONFIG['y_dim'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(input_tensor):
    batch_size, _, x_dim, y_dim = input_tensor.shape
    xs = jnp.arange(x_dim, dtype=jnp.float32) / (x_dim - 1) * 2 - 1
    ys = jnp.arange(y_dim, dtype=jnp.float32) / (y_dim - 1) * 2 - 1
    target = (batch_size, 1, x_dim, y_dim)
    xx = jnp.broadcast_to(xs[None, None, :, None], target).astype(input_tensor.dtype)
    yy = jnp.broadcast_to(ys[None, None, None, :], target).astype(input_tensor.dtype)
    ret = jnp.concatenate([input_tensor, xx, yy], axis=1)
    if CONFIG['with_r']:
        center = CONFIG['r_center']
        rr = jnp.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        ret = jnp.concatenate([ret, rr], axis=1)
    return ret
