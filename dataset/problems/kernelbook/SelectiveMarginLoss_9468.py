"""SelectiveMarginLoss: clamp(pos - neg + margin) * mask, sum, divide by max(sum(mask), 1), scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelectiveMarginLoss_9468',
    'shape': (16, 128, 128, 128),
    'margin': 0.2,
    'loss_weight': 5e-05,
    'clamp_min': 0.0,
    'clamp_max': 1000000.0,
}


def create_inputs(dtype=jnp.float32):
    k_pos, k_neg, k_has = jax.random.split(jax.random.key(0), 3)
    shape = CONFIG['shape']
    pos_samples = jax.random.uniform(k_pos, shape, dtype)
    neg_samples = jax.random.uniform(k_neg, shape, dtype)
    has_sample = jax.random.uniform(k_has, shape, dtype)
    return pos_samples, neg_samples, has_sample


def workload(pos_samples, neg_samples, has_sample):
    margin_diff = jnp.clip(pos_samples - neg_samples + CONFIG['margin'],
                           CONFIG['clamp_min'], CONFIG['clamp_max'])
    num_sample = jnp.maximum(jnp.sum(has_sample), 1.0)
    return CONFIG['loss_weight'] * (jnp.sum(margin_diff * has_sample) / num_sample)
