"""[bf16 variant] MSELoss: squared difference, mean over dim 1, sum, divide by batch size."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MSELoss_11594',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    pred = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return pred, target


def workload(pred, target):
    loss = jnp.mean((target - pred) ** 2, axis=1)
    return jnp.sum(loss) / pred.shape[0]
