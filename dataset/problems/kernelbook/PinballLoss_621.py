"""[bf16 variant] PinballLoss: error = target - pred, max((q-1)*error, q*error), sum over dim 1, mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PinballLoss_621',
    'batch': 16,
    'dim1': 128,
    'dim2': 128,
    'dim3': 128,
    'quantile': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    pred = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return pred, target


def workload(pred, target):
    q = CONFIG['quantile']
    error = target - pred
    losses = jnp.maximum((q - 1.0) * error, q * error)
    return jnp.mean(jnp.sum(losses, axis=1))
