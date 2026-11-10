"""[bf16 variant] Entropy minimization loss: softmax, log_softmax, -sum(p * log p) / batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'EntMinLoss_3887',
    'd0': 16,
    'd1': 64,
    'd2': 128,
    'd3': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['d0'], CONFIG['d1'], CONFIG['d2'], CONFIG['d3'])
    f_x = jax.random.uniform(jax.random.key(0), shape, dtype)
    return (f_x,)


def workload(f_x):
    log_p = jax.nn.log_softmax(f_x, axis=-1)
    p = jnp.exp(log_p)
    return -jnp.sum(p * log_p) / f_x.shape[0]
