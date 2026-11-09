"""BinaryReg: alpha * mean(1 / clamp(|pred - 0.5|, min=0.01))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BinaryReg_3980',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'alpha': 0.1,
    'clamp_min': 0.01,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    pred = jax.random.uniform(key, shape, dtype)
    return (pred,)


def workload(pred):
    diff = jnp.maximum(jnp.abs(pred - 0.5), CONFIG['clamp_min'])
    return CONFIG['alpha'] * jnp.mean(1.0 / diff)
