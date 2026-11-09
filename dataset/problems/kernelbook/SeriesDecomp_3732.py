"""[bf16 variant] SeriesDecomp: edge-replicate pad -> AvgPool1d(kernel_size, stride=1) over time -> residual = x - moving_mean."""
import jax
import jax.numpy as jnp
from jax import lax

# seq_len must stay 2: with even kernel_size=4 the moving mean has length seq_len-1,
# and PyTorch's x - moving_mean only broadcasts when that is 1.
CONFIG = {
    'name': 'SeriesDecomp_3732',
    'batch': 4096,
    'seq_len': 2,
    'channels': 4096,
    'kernel_size': 4,
    'stride': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['channels']), dtype)
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    pad = (k - 1) // 2
    front = jnp.repeat(x[:, 0:1, :], pad, axis=1)
    end = jnp.repeat(x[:, -1:, :], pad, axis=1)
    xp = jnp.concatenate([front, x, end], axis=1)
    window_sum = lax.reduce_window(
        xp, jnp.zeros((), xp.dtype), lax.add,
        window_dimensions=(1, k, 1),
        window_strides=(1, CONFIG['stride'], 1),
        padding='VALID',
    )
    moving_mean = window_sum / k
    res = x - moving_mean
    return res, moving_mean
