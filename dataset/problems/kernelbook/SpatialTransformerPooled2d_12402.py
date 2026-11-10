"""SpatialTransformerPooled2d: clamp grid -> bilinear grid_sample over an avg-pool pyramid -> channel concat -> feature-weighted sum -> bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SpatialTransformerPooled2d_12402',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'outdims': 1024,
    'pool_steps': 1,
    'pool_kern': 2,
    'init_range': 0.1,
    'grid_min': -1.0,
    'grid_max': 1.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_grid, k_feat, k_bias = jax.random.split(jax.random.key(0), 4)
    c = CONFIG['channels']
    outdims = CONFIG['outdims']
    m = CONFIG['pool_steps'] + 1
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype
    )
    grid = jax.random.uniform(
        k_grid,
        (1, outdims, 1, 2),
        dtype,
        minval=-CONFIG['init_range'],
        maxval=CONFIG['init_range'],
    )
    features = jax.random.normal(k_feat, (1, m * c, 1, outdims), dtype) * CONFIG['weight_scale']
    bias = jax.random.normal(k_bias, (outdims,), dtype) * CONFIG['weight_scale']
    return x, grid, features, bias


def _avg_pool(x):
    k = CONFIG['pool_kern']
    window = (1, 1, k, k)
    summed = lax.reduce_window(x, 0.0, lax.add, window, window, 'VALID')
    return summed / (k * k)


def _grid_sample(x, gx, gy):
    # x: (N, c, H, W); gx, gy: (P,) normalized coordinates shared by the batch.
    # Bilinear, zero padding, align_corners=True. Returns (N, c, P).
    H, W = x.shape[2], x.shape[3]
    ix = (gx + 1.0) * 0.5 * (W - 1)
    iy = (gy + 1.0) * 0.5 * (H - 1)
    x0 = jnp.floor(ix)
    y0 = jnp.floor(iy)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    def corner(cx, cy, weight):
        valid = (cx >= 0) & (cx <= W - 1) & (cy >= 0) & (cy <= H - 1)
        xi = jnp.clip(cx, 0, W - 1).astype(jnp.int32)
        yi = jnp.clip(cy, 0, H - 1).astype(jnp.int32)
        return x[:, :, yi, xi] * (weight * valid.astype(weight.dtype))

    return (
        corner(x0, y0, (x1 - ix) * (y1 - iy))
        + corner(x0, y1, (x1 - ix) * (iy - y0))
        + corner(x1, y0, (ix - x0) * (y1 - iy))
        + corner(x1, y1, (ix - x0) * (iy - y0))
    )


def workload(x, grid, features, bias):
    N, c = x.shape[0], x.shape[1]
    outdims = grid.shape[1]
    m = features.shape[1] // c

    grid = jnp.clip(grid, CONFIG['grid_min'], CONFIG['grid_max'])
    gx = grid[0, :, 0, 0]
    gy = grid[0, :, 0, 1]
    feat = features.reshape(1, m * c, outdims)

    pools = [_grid_sample(x, gx, gy)]
    for _ in range(m - 1):
        x = _avg_pool(x)
        pools.append(_grid_sample(x, gx, gy))
    y = jnp.concatenate(pools, axis=1)

    y = jnp.sum(y * feat, axis=1).reshape(N, outdims)
    return y + bias
