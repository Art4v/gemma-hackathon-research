"""Pearson correlation: global-mean centering of two tensors -> sum of products -> divide by product of root-sum-squares."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PLCCLoss_6074',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key_input, key_target = jax.random.split(jax.random.key(0))
    input_ = jax.random.uniform(key_input, shape, dtype)
    target = jax.random.uniform(key_target, shape, dtype)
    return input_, target


def workload(input_, target):
    input0 = input_ - jnp.mean(input_)
    target0 = target - jnp.mean(target)
    return jnp.sum(input0 * target0) / (
        jnp.sqrt(jnp.sum(input0 ** 2)) * jnp.sqrt(jnp.sum(target0 ** 2))
    )
