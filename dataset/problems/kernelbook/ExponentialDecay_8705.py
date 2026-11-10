"""[bf16 variant] Exponential decay recurrence over axis 0: state = x[t] * (1 - alpha) + state * alpha, stacked via lax.scan."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ExponentialDecay_8705',
    'time_steps': 32,
    'dim_1': 128,
    'dim_2': 128,
    'dim_3': 128,
    'alpha': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['time_steps'], CONFIG['dim_1'], CONFIG['dim_2'], CONFIG['dim_3'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    alpha = CONFIG['alpha']

    def step(state, x_t):
        new_state = x_t * (1 - alpha) + state * alpha
        return new_state, new_state

    state, out = lax.scan(step, x[0], x)
    return out, state
