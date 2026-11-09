"""Distance-weighted MSE loss: (output - target)^2 * weight(target, borders), mean-reduced."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DistanceWeightedMSELoss_13604',
    'batch': 8,
    'channels': 16,
    'height': 512,
    'width': 512,
    'fg_weight': 100.0,
    'mask_borders': 40,
}


def create_inputs(dtype=jnp.float32):
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    output = jax.random.uniform(k_out, shape, dtype)
    # Signed-distance-like targets so the target <= 0 (foreground) branch is exercised.
    target = jax.random.uniform(k_tgt, shape, dtype, minval=-1.0, maxval=1.0)
    return output, target


def workload(output, target):
    o = CONFIG['mask_borders']
    h, w = target.shape[-2], target.shape[-1]
    mse = (output - target) ** 2
    weight = jnp.where(target <= 0, jnp.asarray(CONFIG['fg_weight'], target.dtype),
                       jnp.asarray(1.0, target.dtype))
    rows = lax.broadcasted_iota(jnp.int32, (h, w), 0)
    cols = lax.broadcasted_iota(jnp.int32, (h, w), 1)
    # Same as torch slices [:o, :o] and [h-o:, w-o:] (negative start clamps to 0).
    border = ((rows < o) & (cols < o)) | ((rows >= h - o) & (cols >= w - o))
    weight = jnp.where(border, jnp.zeros_like(weight), weight)
    return jnp.mean(weight * mse)
