"""[bf16 variant] AEBatch: abs(sum(density_map, dims 1,2,3) - gt_num) with PyTorch trailing-axis broadcasting."""
import jax
import jax.numpy as jnp

# width must equal batch: the (B,) sum broadcasts against gt_num's last axis, as in PyTorch.
CONFIG = {
    'name': 'AEBatch_18180',
    'batch': 128,
    'channels': 8,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    estimated_density_map = jax.random.uniform(k1, shape, dtype)
    gt_num = jax.random.uniform(k2, shape, dtype)
    return estimated_density_map, gt_num


def workload(estimated_density_map, gt_num):
    s = jnp.sum(estimated_density_map, axis=(1, 2, 3))
    return jnp.abs(s - gt_num)
