"""AddCoords: concat normalized x/y coordinate channels (optional radius channel) onto an NCHW tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AddCoords_14011',
    'batch': 16,
    'channels': 64,
    'x_dim': 256,
    'y_dim': 256,
    'with_r': False,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['x_dim'], CONFIG['y_dim']), dtype)
    return (x,)


def workload(x):
    b, _, x_dim, y_dim = x.shape
    xs = jnp.arange(x_dim, dtype=jnp.float32) / (x_dim - 1) * 2 - 1
    ys = jnp.arange(y_dim, dtype=jnp.float32) / (y_dim - 1) * 2 - 1
    # xx varies along dim 2 (x_dim), yy along dim 3 (y_dim), matching the torch repeat/transpose chain.
    xx = jnp.broadcast_to(xs[None, None, :, None], (b, 1, x_dim, y_dim)).astype(x.dtype)
    yy = jnp.broadcast_to(ys[None, None, None, :], (b, 1, x_dim, y_dim)).astype(x.dtype)
    ret = jnp.concatenate([x, xx, yy], axis=1)
    if CONFIG['with_r']:
        rr = jnp.sqrt(jnp.square(xx - 0.5) + jnp.square(yy - 0.5))
        ret = jnp.concatenate([ret, rr], axis=1)
    return ret
