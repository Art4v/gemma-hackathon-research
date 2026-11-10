"""DistillMSE: softmax(axis=1) on student and teacher logits followed by mean squared error."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DistillMSE_7688',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_s, k_t = jax.random.split(key)
    y_s = jax.random.uniform(k_s, shape, dtype)
    y_t = jax.random.uniform(k_t, shape, dtype)
    return y_s, y_t


def workload(y_s, y_t):
    axis = CONFIG['softmax_axis']
    p_s = jax.nn.softmax(y_s, axis=axis)
    p_t = jax.nn.softmax(y_t, axis=axis)
    return jnp.mean(jnp.square(p_s - p_t))
