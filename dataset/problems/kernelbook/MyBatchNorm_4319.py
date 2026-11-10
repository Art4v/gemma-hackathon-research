"""MyBatchNorm: var_mean over axis 0 (ddof=1) -> standardize -> gamma * z + beta on trailing axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MyBatchNorm_4319',
    'batch': 16,
    'dim1': 128,
    'dim2': 128,
    'dim3': 128,
    'size': 128,
    'epsilon': 1e-05,
    'stat_axis': 0,
    'ddof': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    x = jax.random.uniform(key, shape, dtype)
    gamma = jnp.ones((CONFIG['size'],), dtype)
    beta = jnp.zeros((CONFIG['size'],), dtype)
    return x, gamma, beta


def workload(x, gamma, beta):
    axis = CONFIG['stat_axis']
    var = jnp.var(x, axis=axis, ddof=CONFIG['ddof'])
    mean = jnp.mean(x, axis=axis)
    z = (x - mean) / jnp.sqrt(var + CONFIG['epsilon'])
    return gamma * z + beta
