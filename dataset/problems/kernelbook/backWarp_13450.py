"""Backward warp: flow-offset mesh grid, normalize, bilinear grid_sample (border padding, align_corners=False)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'backWarp_13450',
    'batch': 8,
    'channels': 4,
    'flow_channels': 4,
    'height': 1024,
    'width': 1024,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_img, k_flow = jax.random.split(key)
    img = jax.random.uniform(
        k_img,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    flow = jax.random.uniform(
        k_flow,
        (CONFIG['batch'], CONFIG['flow_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return img, flow


def _gather(im, iy, ix):
    return im[:, iy, ix]


def workload(img, flow):
    _, _, H, W = img.shape
    dtype = img.dtype

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]
    x = jnp.arange(W, dtype=dtype)[None, None, :] + u
    y = jnp.arange(H, dtype=dtype)[None, :, None] + v

    x = 2 * (x / W - 0.5)
    y = 2 * (y / H - 0.5)

    px = ((x + 1) * W - 1) * 0.5
    py = ((y + 1) * H - 1) * 0.5
    px = jnp.clip(px, 0, W - 1)
    py = jnp.clip(py, 0, H - 1)

    x0 = jnp.floor(px)
    y0 = jnp.floor(py)
    wx = (px - x0)[:, None, :, :]
    wy = (py - y0)[:, None, :, :]

    ix0 = x0.astype(jnp.int32)
    iy0 = y0.astype(jnp.int32)
    ix1 = jnp.minimum(ix0 + 1, W - 1)
    iy1 = jnp.minimum(iy0 + 1, H - 1)

    gather = jax.vmap(_gather)
    v00 = gather(img, iy0, ix0)
    v01 = gather(img, iy0, ix1)
    v10 = gather(img, iy1, ix0)
    v11 = gather(img, iy1, ix1)

    top = v00 * (1 - wx) + v01 * wx
    bottom = v10 * (1 - wx) + v11 * wx
    return top * (1 - wy) + bottom * wy
