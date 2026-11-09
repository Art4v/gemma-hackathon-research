"""[bf16 variant] Moving average: edge-replicate pad along time, then AvgPool1d over time."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moving_avg_16583',
    'batch': 16,
    'seq_len': 2048,
    'channels': 1024,
    'kernel_size': 4,
    'stride': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['channels']), dtype)
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    pad = (k - 1) // 2
    front = jnp.repeat(x[:, 0:1, :], pad, axis=1)
    end = jnp.repeat(x[:, -1:, :], pad, axis=1)
    xp = jnp.concatenate([front, x, end], axis=1)
    summed = lax.reduce_window(xp, jnp.array(0, xp.dtype), lax.add,
                               (1, k, 1), (1, s, 1), 'VALID')
    return summed / k
