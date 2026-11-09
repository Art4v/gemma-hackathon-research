"""LayerScaling1d: per-group x / sqrt(mean(x^2) + eps) over channels (RMS scaling)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerScaling1d_13508',
    'batch': 8192,
    'channels': 4096,
    'group_size': -1,  # -1 means one group spanning all channels (PyTorch default)
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['channels']), dtype=dtype)
    return (x,)


def workload(x):
    n, c = x.shape
    g = c if CONFIG['group_size'] == -1 else CONFIG['group_size']
    tmp = x.reshape(n, c // g, g)
    moment2 = jnp.mean(tmp * tmp, axis=2, keepdims=True)
    out = tmp / jnp.sqrt(moment2 + CONFIG['eps'])
    return out.reshape(n, c)
