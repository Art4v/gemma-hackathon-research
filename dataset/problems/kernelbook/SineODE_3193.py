"""SineODE right-hand side: 2*y/t + t**4*sin(2t) - t**2 + 4*t**3 (elementwise)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SineODE_3193',
    'shape': (64, 512, 1024),
    't_min': 0.01,  # keeps 2*y/t finite; uniform [0,1) hits exact 0 a few times at this size
}


def create_inputs(dtype=jnp.float32):
    k_t, k_y = jax.random.split(jax.random.key(0))
    shape = CONFIG['shape']
    t = jax.random.uniform(k_t, shape, dtype, minval=CONFIG['t_min'], maxval=1.0)
    y = jax.random.uniform(k_y, shape, dtype)
    return t, y


def workload(t, y):
    return 2 * y / t + t ** 4 * jnp.sin(2 * t) - t ** 2 + 4 * t ** 3
