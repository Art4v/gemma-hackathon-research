"""MyMaxPool1dPadSame: zero SAME-padding on the last axis followed by MaxPool1d(kernel_size=K, stride=K)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MyMaxPool1dPadSame_2963',
    'rows': 1024,
    'length': 65536,
    'kernel_size': 4,
    'stride_size': 1,  # only used for the SAME-padding arithmetic; pool stride = kernel_size
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['rows'], CONFIG['length']), dtype)
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride_size']
    in_dim = x.shape[-1]
    out_dim = (in_dim + s - 1) // s
    p = max(0, (out_dim - 1) * s + k - in_dim)
    pad_left = p // 2
    pad_right = p - pad_left
    pad_cfg = [(0, 0)] * (x.ndim - 1) + [(pad_left, pad_right)]
    net = jnp.pad(x, pad_cfg, mode='constant', constant_values=0)
    # nn.MaxPool1d(kernel_size=k) defaults stride to k, no padding, floor mode.
    window = (1,) * (x.ndim - 1) + (k,)
    return lax.reduce_window(
        net, jnp.array(-jnp.inf, dtype=net.dtype), lax.max, window, window, 'VALID'
    )
