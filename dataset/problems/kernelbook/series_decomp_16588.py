"""[bf16 variant] series_decomp: edge-repeat padding -> moving average (sum/k, VALID) over time axis -> residual and trend."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'series_decomp_16588',
    'batch': 2048,
    'seq_len': 2,  # kernel_size=4 yields trend length T-1; x - trend only broadcasts for T=2
    'channels': 8192,
    'kernel_size': 4,
    'stride': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['seq_len'], CONFIG['channels']),
        dtype,
    )
    return (x,)


def workload(x):
    kernel_size = CONFIG['kernel_size']
    stride = CONFIG['stride']
    pad = (kernel_size - 1) // 2
    front = jnp.repeat(x[:, :1, :], pad, axis=1)
    end = jnp.repeat(x[:, -1:, :], pad, axis=1)
    padded = jnp.concatenate([front, x, end], axis=1)
    summed = lax.reduce_window(
        padded,
        0.0,
        lax.add,
        window_dimensions=(1, kernel_size, 1),
        window_strides=(1, stride, 1),
        padding='VALID',
    )
    moving_mean = summed / kernel_size
    res = x - moving_mean
    return res, moving_mean
