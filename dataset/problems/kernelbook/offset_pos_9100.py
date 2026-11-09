"""[bf16 variant] offset_pos: label-weighted SmoothL1 (beta=1) summed and normalized by max(1, sum(weight))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'offset_pos_9100',
    'batch': 16,
    'pred_channels': 2,
    'label_channels': 4,
    'weight_channel': 2,
    'height': 512,
    'width': 512,
    'beta': 1.0,
    'min_norm': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    b, h, w = CONFIG['batch'], CONFIG['height'], CONFIG['width']
    offset_pred = jax.random.uniform(k1, (b, CONFIG['pred_channels'], h, w), dtype)
    offset_label = jax.random.uniform(k2, (b, CONFIG['label_channels'], h, w), dtype)
    return offset_pred, offset_label


def workload(offset_pred, offset_label):
    nc = offset_pred.shape[1]
    wc = CONFIG['weight_channel']
    beta = CONFIG['beta']
    weight = offset_label[:, wc:wc + 1, :, :]
    d = jnp.abs(offset_pred - offset_label[:, :nc, :, :])
    sl1 = jnp.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta)
    total = jnp.sum(weight * sl1)
    return total / jnp.maximum(CONFIG['min_norm'], jnp.sum(weight))
