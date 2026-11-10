"""Backprojection: inv_K[:3,:3] @ pixel_grid, scaled by depth, concat homogeneous ones."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Backprojection_10913',
    'batch_size': 16,
    'height': 1024,
    'width': 2048,
}


def create_inputs(dtype=jnp.float32):
    n, h, w = CONFIG['batch_size'], CONFIG['height'], CONFIG['width']
    k_depth, k_invk = jax.random.split(jax.random.key(0))
    depth = jax.random.uniform(k_depth, (n, h, w), dtype)
    inv_K = jax.random.uniform(k_invk, (n, 4, 4), dtype)
    return depth, inv_K


def workload(depth, inv_K):
    n = depth.shape[0]
    h, w = depth.shape[-2], depth.shape[-1]
    dtype = depth.dtype
    # Pixel grid matching torch.meshgrid 'ij' flattened row-major: x = col, y = row.
    yy, xx = jnp.meshgrid(jnp.arange(h, dtype=dtype), jnp.arange(w, dtype=dtype), indexing='ij')
    xx = xx.reshape(1, 1, -1)
    yy = yy.reshape(1, 1, -1)
    k = inv_K[:, :3, :3]
    # 3x3 @ [x; y; 1] done elementwise to keep full f32 precision.
    cam_p_norm = k[:, :, 0:1] * xx + k[:, :, 1:2] * yy + k[:, :, 2:3]
    cam_p_euc = depth.reshape(n, 1, -1) * cam_p_norm
    ones = jnp.ones((n, 1, h * w), dtype)
    return jnp.concatenate([cam_p_euc, ones], axis=1)
