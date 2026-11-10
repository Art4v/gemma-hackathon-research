"""Pairwise L2 distance (eps on difference) -> clamped threshold/margin hinge terms -> label-weighted squared sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ConstractiveThresholdHingeLoss_15808',
    'batch': 8,
    'height': 128,
    'width': 128,
    'vector_dim': 128,
    'label_extra_dim': 128,
    'hinge_threshold': 0.0,
    'margin': 2.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_t0, k_t1, k_label = jax.random.split(key, 3)
    vec_shape = (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['vector_dim'])
    # The distance drops the last axis, and the label aligns with it on trailing dimensions.
    label_shape = (CONFIG['label_extra_dim'], CONFIG['batch'], CONFIG['height'], CONFIG['width'])
    out_vec_t0 = jax.random.uniform(k_t0, vec_shape, dtype)
    out_vec_t1 = jax.random.uniform(k_t1, vec_shape, dtype)
    label = jax.random.uniform(k_label, label_shape, dtype)
    return out_vec_t0, out_vec_t1, label


def workload(out_vec_t0, out_vec_t1, label):
    diff = out_vec_t0 - out_vec_t1 + CONFIG['eps']
    distance = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1))
    similar_pair = jnp.maximum(distance - CONFIG['hinge_threshold'], 0.0)
    dissimilar_pair = jnp.maximum(CONFIG['margin'] - distance, 0.0)
    return jnp.sum((1 - label) * jnp.square(similar_pair) + label * jnp.square(dissimilar_pair))
