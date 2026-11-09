"""[bf16 variant] MaskedSoftmax (mask=None): logit - max(logit, dim) followed by softmax over dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MaskedSoftmax_14127',
    'shape': (8, 8, 32, 64, 512),
    'dim': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    logit = jax.random.uniform(key, CONFIG['shape'], dtype)
    return (logit,)


def workload(logit):
    dim = CONFIG['dim']
    shifted = logit - jnp.max(logit, axis=dim, keepdims=True)
    return jax.nn.softmax(shifted, axis=dim)
