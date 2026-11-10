"""[bf16 variant] SinenetComponent: a * sin(i * f * t + phi) * x summed over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SinenetComponent_7273',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'param_size': 1,
    'time_len': 4,
    'i': 4,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    param_shape = (CONFIG['param_size'],)
    key = jax.random.key(0)
    k_x, k_f, k_t, k_a, k_phi = jax.random.split(key, 5)
    x = jax.random.uniform(k_x, shape, dtype)
    f = jax.random.uniform(k_f, shape, dtype)
    t = jax.random.uniform(k_t, shape, dtype)
    a = jax.random.normal(k_a, param_shape, dtype) * CONFIG['weight_scale']
    phi = jax.random.normal(k_phi, param_shape, dtype) * CONFIG['weight_scale']
    return x, f, t, a, phi


def workload(x, f, t, a, phi):
    i_f = CONFIG['i'] * f
    i_f_t = i_f * t
    deg = i_f_t + phi
    s = jnp.sin(deg)
    w = a * s
    h_sbt = w * x
    return jnp.sum(h_sbt, axis=-1, keepdims=False)
