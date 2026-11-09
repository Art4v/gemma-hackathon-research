"""GetSegPred: round/scale point cloud to voxel indices, fix -1 -> 0, transpose(1,4), per-batch gather."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GetSegPred_10461',
    'batch': 8,
    'channels': 128,
    'grid': 32,
    'num_points': 65536,
    'coord_dim': 3,
    'scale': 1.0,
    'offset': 0.501,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    b, c, g = CONFIG['batch'], CONFIG['channels'], CONFIG['grid']
    segs = jax.random.uniform(k1, (b, c, g, g, g), dtype)
    ptcloud = jax.random.uniform(k2, (b, CONFIG['num_points'], CONFIG['coord_dim']), dtype)
    return segs, ptcloud


def workload(segs, ptcloud):
    # Module stores scale // 2 (floor division on a float), so scale=1.0 -> 0.0.
    half_scale = CONFIG['scale'] // 2
    t = jnp.round((ptcloud + 1) * half_scale - CONFIG['offset']).astype(jnp.int32)
    t = jnp.where(t == -1, 0, t)
    segs_t = jnp.swapaxes(segs, 1, 4)
    i0, i1, i2 = t[:, :, 0], t[:, :, 1], t[:, :, 2]
    bidx = jnp.arange(segs.shape[0]).reshape((segs.shape[0],) + (1,) * (i0.ndim - 1))
    return segs_t[bidx, i0, i1, i2]
