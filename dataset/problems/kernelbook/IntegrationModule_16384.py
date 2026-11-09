"""[bf16 variant] IntegrationModule: clamp((iou - min_iou)/(1 - min_iou), 0, 1) * wmax weights, broadcast enhance*we - reduce*wr."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'IntegrationModule_16384',
    'overlap_a': 2,
    'overlap_b': 2,
    'overlap_c': 2,
    'shared_d': 8,
    'feat_e': 128,
    'feat_f': 128,
    'feat_g': 128,
    'min_iou': 0.2,
    'enhance_weight_max': 1.0,
    'reduce_weight_max': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    feat_shape = (c['shared_d'], c['feat_e'], c['feat_f'], c['feat_g'])
    ov_shape = (c['overlap_a'], c['overlap_b'], c['overlap_c'], c['shared_d'])
    enhance_feature = jax.random.uniform(k1, feat_shape, dtype)
    reduce_feature = jax.random.uniform(k2, feat_shape, dtype)
    overlaps = jax.random.uniform(k3, ov_shape, dtype)
    return enhance_feature, reduce_feature, overlaps


def _compute_weight(ious, weight_max, iou_min):
    x = (ious - iou_min) / (1.0 - iou_min)
    return weight_max * jnp.minimum(jnp.maximum(x, jnp.zeros_like(ious)), jnp.ones_like(ious))


def workload(enhance_feature, reduce_feature, overlaps):
    ew = _compute_weight(overlaps, CONFIG['enhance_weight_max'], CONFIG['min_iou'])
    rw = _compute_weight(overlaps, CONFIG['reduce_weight_max'], CONFIG['min_iou'])
    # Same broadcast as torch's triple unsqueeze(-1): result is overlaps.shape[:-1] + feature.shape.
    return enhance_feature * ew[..., None, None, None] - reduce_feature * rw[..., None, None, None]
