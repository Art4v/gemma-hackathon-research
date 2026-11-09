"""MaxPoolTrinary: max over the left, middle and right thirds of the length axis, then concat."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MaxPoolTrinary_17131',
    'batch': 16,
    'length': 384,
    'dim1': 32,
    'dim2': 256,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    states = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['length'], CONFIG['dim1'], CONFIG['dim2']), dtype)
    return (states,)


def workload(states):
    length = states.shape[1]
    side = (length + 1) // 3
    left = jnp.max(states[:, :side], axis=1, keepdims=True)
    mid = jnp.max(states[:, side:length - side], axis=1, keepdims=True)
    right = jnp.max(states[:, length - side:], axis=1, keepdims=True)
    return jnp.concatenate([left, mid, right], axis=1)
