"""[bf16 variant] SineODE right-hand side: 2*y/t + t^4*sin(2*t) - t^2 + 4*t^3 (elementwise)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SineODE_1157',
    'shape': (8, 128, 128, 128),
    't_min': 1e-3,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_t, k_y = jax.random.split(key)
    t = jax.random.uniform(k_t, CONFIG['shape'], dtype, minval=CONFIG['t_min'], maxval=1.0)
    y = jax.random.uniform(k_y, CONFIG['shape'], dtype)
    return t, y


def workload(t, y):
    return 2 * y / t + t ** 4 * jnp.sin(2 * t) - t ** 2 + 4 * t ** 3
