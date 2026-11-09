"""[bf16 variant] Arc2: concat([cos(pi*t/2), 0*(1-t-cos), 0*(t-sin), sin(pi*t/2)]) along dim 0."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Arc2_1502',
    'num_bends': 4,
    'shape': (8, 256, 256, 128),
}


def create_inputs(dtype=jnp.bfloat16):
    t = jax.random.uniform(jax.random.key(0), CONFIG['shape'], dtype)
    return (t,)


def workload(t):
    a = jnp.cos(jnp.pi * t / 2)
    b = jnp.sin(jnp.pi * t / 2)
    return jnp.concatenate([a, 0.0 * (1.0 - t - a), 0.0 * (t - b), b], axis=0)
