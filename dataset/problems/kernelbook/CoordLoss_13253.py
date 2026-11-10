"""[bf16 variant] CoordLoss: abs(coord_out - coord_gt) * valid (is_3D=None path)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CoordLoss_13253',
    'batch': 16,
    'dim1': 128,
    'dim2': 128,
    'dim3': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    coord_out = jax.random.uniform(k1, shape, dtype)
    coord_gt = jax.random.uniform(k2, shape, dtype)
    valid = jax.random.uniform(k3, shape, dtype)
    return coord_out, coord_gt, valid


def workload(coord_out, coord_gt, valid):
    # ponytail: is_3D branch omitted, get_inputs never passes it.
    return jnp.abs(coord_out - coord_gt) * valid
