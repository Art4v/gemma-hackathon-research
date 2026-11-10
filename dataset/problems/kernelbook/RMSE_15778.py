"""Masked RMSE: mask=(target>thr), per-sample sqrt(sum((t*m - o*m)^2) / sum(m))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RMSE_15778',
    'batch': 16,
    'channels': 8,
    'height': 512,
    'width': 512,
    'threshold': 0.001,
}


def create_inputs(dtype=jnp.float32):
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    outputs = jax.random.uniform(k_out, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return outputs, target


def workload(outputs, target):
    val_pixels = (target > CONFIG['threshold']).astype(outputs.dtype)
    err = (target * val_pixels - outputs * val_pixels) ** 2
    b = err.shape[0]
    loss = jnp.sum(err.reshape(b, 1, -1), axis=-1, keepdims=True)
    cnt = jnp.sum(val_pixels.reshape(b, 1, -1), axis=-1, keepdims=True)
    return jnp.sqrt(loss / cnt)
