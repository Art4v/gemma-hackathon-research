"""[bf16 variant] AttentionCrossEntropy: -mean(target * log_softmax(input, axis=1))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentionCrossEntropy_8444',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return x, target


def workload(x, target):
    logp = jax.nn.log_softmax(x, axis=CONFIG['softmax_axis'])
    return -jnp.mean(target.astype(jnp.float32) * logp)
