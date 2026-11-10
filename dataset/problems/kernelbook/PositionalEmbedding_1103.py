"""Sinusoidal positional embedding: arange -> exp frequency scale -> sin/cos concat -> optional zero pad -> broadcast add."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PositionalEmbedding_1103',
    'batch': 32,
    'length': 1024,
    'channels': 1024,
    'log_max_period': 9.210340371976184,  # math.log(10000.0) as a double
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    inputs = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['length'], CONFIG['channels']), dtype
    )
    return (inputs,)


def workload(inputs):
    if inputs.ndim != 3:
        raise ValueError('The rank of input must be 3.')
    length = inputs.shape[1]
    channels = inputs.shape[2]
    half_dim = channels // 2
    positions = jnp.arange(length, dtype=inputs.dtype)
    dimensions = jnp.arange(half_dim, dtype=inputs.dtype)
    scale = CONFIG['log_max_period'] / float(half_dim - 1)
    dimensions = jnp.exp(dimensions * (-scale))
    scaled_time = positions[:, None] * dimensions[None, :]
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)
    if channels % 2 == 1:
        pad = jnp.zeros((signal.shape[0], 1), dtype=inputs.dtype)
        signal = jnp.concatenate([signal, pad], axis=1)
    return inputs + jnp.reshape(signal, (1, -1, channels))
