"""[bf16 variant] Series decomposition: edge-replicated multi-kernel moving averages, softmax(Linear) weighted sum, residual."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'series_decomp_multi_17652',
    'batch': 64,
    'seq_len': 1024,
    'channels': 1024,
    'kernel_sizes': (4, 4),
    'stride': 1,
    'linear_in': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    num_kernels = len(CONFIG['kernel_sizes'])
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['channels']), dtype
    )
    w = jax.random.normal(k_w, (CONFIG['linear_in'], num_kernels), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (num_kernels,), dtype) * CONFIG['weight_scale']
    return x, w, b


def _moving_avg(x, kernel):
    end = (kernel - 1) // 2
    front = kernel - 1 - end
    padded = jnp.pad(x, ((0, 0), (front, end), (0, 0)), mode='edge')
    summed = lax.reduce_window(
        padded,
        jnp.zeros((), x.dtype),
        lax.add,
        (1, kernel, 1),
        (1, CONFIG['stride'], 1),
        'VALID',
    )
    return summed / kernel


def workload(x, w, b):
    trends = jnp.stack([_moving_avg(x, k) for k in CONFIG['kernel_sizes']], axis=-1)
    weights = jax.nn.softmax(x[..., None] @ w + b, axis=-1)
    moving_mean = jnp.sum(trends * weights, axis=-1)
    res = x - moving_mean
    return res, moving_mean
