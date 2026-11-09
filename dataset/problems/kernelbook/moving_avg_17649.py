"""Edge-replicate pad along time, then AvgPool1d over time (moving average trend block)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moving_avg_17649',
    'batch': 16,
    'seq_len': 2048,
    'channels': 1024,
    'kernel_size': 4,
    'stride': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['channels']), dtype=dtype
    )
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    n_end = (k - 1) // 2
    n_front = k - 1 - n_end
    front = jnp.repeat(x[:, 0:1, :], n_front, axis=1)
    end = jnp.repeat(x[:, -1:, :], n_end, axis=1)
    xp = jnp.concatenate([front, x, end], axis=1)
    summed = lax.reduce_window(
        xp, jnp.array(0, xp.dtype), lax.add,
        window_dimensions=(1, k, 1),
        window_strides=(1, s, 1),
        padding='VALID',
    )
    return summed / k
