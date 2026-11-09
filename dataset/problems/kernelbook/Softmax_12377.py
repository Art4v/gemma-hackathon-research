"""[bf16 variant] Unstabilized softmax over dim 0: exp(x) / sum(exp(x), axis=0)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Softmax_12377',
    'd0': 16,
    'd1': 256,
    'd2': 128,
    'd3': 128,
    'axis': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['d0'], CONFIG['d1'], CONFIG['d2'], CONFIG['d3'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    # ponytail: no max-subtraction, matches the PyTorch module exactly; inputs in [0,1) keep exp finite.
    y = jnp.exp(x)
    return y / jnp.sum(y, axis=CONFIG['axis'])
