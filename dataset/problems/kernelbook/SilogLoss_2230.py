"""[bf16 variant] SilogLoss: log(pred*ratio) - log(gt*ratio) -> mean(d^2) - ratio2*mean(d)^2 -> sqrt -> scale by ratio."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SilogLoss_2230',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'ratio': 10.0,
    'ratio2': 0.85,
    'minval': 1e-3,
    'maxval': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_pred, k_gt = jax.random.split(jax.random.key(0))
    pred = jax.random.uniform(k_pred, shape, dtype, minval=CONFIG['minval'], maxval=CONFIG['maxval'])
    gt = jax.random.uniform(k_gt, shape, dtype, minval=CONFIG['minval'], maxval=CONFIG['maxval'])
    return pred, gt


def workload(pred, gt):
    ratio = CONFIG['ratio']
    ratio2 = CONFIG['ratio2']
    log_diff = jnp.log(pred * ratio) - jnp.log(gt * ratio)
    silog1 = jnp.mean(log_diff ** 2)
    silog2 = ratio2 * jnp.mean(log_diff) ** 2
    return jnp.sqrt(silog1 - silog2) * ratio
