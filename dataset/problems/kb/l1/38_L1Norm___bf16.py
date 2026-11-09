"""[bf16 variant] L1 normalization: x / mean(abs(x), dim=1, keepdim=True)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '38_L1Norm_',
    'batch': 1024,
    'dim': 65536,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    return x / jnp.mean(jnp.abs(x), axis=1, keepdims=True)
