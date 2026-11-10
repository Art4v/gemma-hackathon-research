"""[bf16 variant] LocalNorm2d: reflect pad -> 33x33 box average of x and x*x -> local z-score -> clamp."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LocalNorm2d_16320',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 33,
    'stride': 1,
    'eps': 1e-10,
    'clamp_min': -6.0,
    'clamp_max': 6.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def _reflect_avg_pool(x):
    ks = CONFIG['kernel_size']
    st = CONFIG['stride']
    pd = ks // 2
    padded = jnp.pad(x, ((0, 0), (0, 0), (pd, pd), (pd, pd)), mode='reflect')
    summed = lax.reduce_window(padded, 0.0, lax.add, (1, 1, ks, 1), (1, 1, st, 1), 'VALID')
    summed = lax.reduce_window(summed, 0.0, lax.add, (1, 1, 1, ks), (1, 1, 1, st), 'VALID')
    return summed / (ks * ks)


def workload(x):
    mean = _reflect_avg_pool(x)
    variance = jnp.abs(_reflect_avg_pool(x * x) - mean * mean)
    normed = (x - mean) / (jnp.sqrt(variance) + CONFIG['eps'])
    return jnp.clip(normed, CONFIG['clamp_min'], CONFIG['clamp_max'])
