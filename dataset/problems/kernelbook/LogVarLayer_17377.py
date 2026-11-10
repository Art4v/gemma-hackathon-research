"""LogVarLayer: log(clamp(var(x, dim, unbiased, keepdim), 1e-6, 1e6))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LogVarLayer_17377',
    'shape': (8, 32, 64, 32, 128),
    'dim': 4,
    'ddof': 1,
    'clamp_min': 1e-06,
    'clamp_max': 1000000.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, CONFIG['shape'], dtype=dtype)
    return (x,)


def workload(x):
    v = jnp.var(x, axis=CONFIG['dim'], ddof=CONFIG['ddof'], keepdims=True)
    return jnp.log(jnp.clip(v, CONFIG['clamp_min'], CONFIG['clamp_max']))
