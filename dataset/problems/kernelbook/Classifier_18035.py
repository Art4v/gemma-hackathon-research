"""Classifier_18035: concat(di_z, ds_z, dim=1) -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Classifier_18035',
    'batch': 8192,
    'feature_dim': 2048,
    'classes': 1024,
}


def create_inputs(dtype=jnp.float32):
    k_a, k_b, k_w, k_bias = jax.random.split(jax.random.key(0), 4)
    b, f, c = CONFIG['batch'], CONFIG['feature_dim'], CONFIG['classes']
    di_z = jax.random.uniform(k_a, (b, f), dtype)
    ds_z = jax.random.uniform(k_b, (b, f), dtype)
    w = jax.random.normal(k_w, (2 * f, c), dtype) * 0.02
    bias = jax.random.normal(k_bias, (c,), dtype) * 0.02
    return di_z, ds_z, w, bias


def workload(di_z, ds_z, w, bias):
    z = jnp.concatenate((di_z, ds_z), axis=1)
    return z @ w + bias
