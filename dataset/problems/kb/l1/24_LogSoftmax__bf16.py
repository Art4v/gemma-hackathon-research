"""[bf16 variant] LogSoftmax along dim=1 of a (batch, dim) tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '24_LogSoftmax',
    'batch_size': 1024,
    'dim': 32768,
    'axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    axis = CONFIG['axis']
    x_max = jnp.max(x, axis=axis, keepdims=True)
    shifted = x - jax.lax.stop_gradient(x_max)
    lse = jnp.log(jnp.sum(jnp.exp(shifted), axis=axis, keepdims=True))
    return shifted - lse
