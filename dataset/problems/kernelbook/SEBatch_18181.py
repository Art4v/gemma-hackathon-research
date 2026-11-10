"""SEBatch: sum over (C,H,W), subtract gt_num with broadcasting, square element-wise."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SEBatch_18181',
    'batch': 128,
    'channels': 16,
    'height': 128,
    'width': 128,
    'gt_batch': 128,
    'gt_channels': 16,
    'gt_height': 128,
    'gt_width': 128,  # must equal 'batch': the (B,) sum broadcasts against gt_num's last axis
    'sum_axes': (1, 2, 3),
    'exponent': 2,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_map, k_gt = jax.random.split(key)
    estimated_density_map = jax.random.uniform(
        k_map,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    gt_num = jax.random.uniform(
        k_gt,
        (CONFIG['gt_batch'], CONFIG['gt_channels'], CONFIG['gt_height'], CONFIG['gt_width']),
        dtype,
    )
    return estimated_density_map, gt_num


def workload(estimated_density_map, gt_num):
    summed = jnp.sum(estimated_density_map, axis=CONFIG['sum_axes'])
    return jnp.power(summed - gt_num, CONFIG['exponent'])
