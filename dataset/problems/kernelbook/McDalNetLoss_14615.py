"""[bf16 variant] McDalNetLoss (L1): softmax(x1, axis=1) and softmax(x2, axis=1), then mean |p1 - p2|."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'McDalNetLoss_14615',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    input1 = jax.random.uniform(k1, shape, dtype)
    input2 = jax.random.uniform(k2, shape, dtype)
    return input1, input2


def workload(input1, input2):
    axis = CONFIG['softmax_axis']
    prob_s = jax.nn.softmax(input1, axis=axis)
    prob_t = jax.nn.softmax(input2, axis=axis)
    return jnp.mean(jnp.abs(prob_s - prob_t))
