"""[bf16 variant] LayerNorm over the last axis: mean, std (ddof=1), normalize, scale by a, shift by b."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_17385',
    'batch': 8,
    'length': 128,
    'channels': 128,
    'input_size': 256,
    'eps': 1e-10,
    'ddof': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['length'], CONFIG['channels'], CONFIG['input_size'])
    x = jax.random.uniform(key, shape, dtype)
    a = jnp.ones((CONFIG['input_size'],), dtype)
    b = jnp.zeros((CONFIG['input_size'],), dtype)
    return x, a, b


def workload(x, a, b):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    sigma = jnp.std(x, axis=-1, keepdims=True, ddof=CONFIG['ddof'])
    output = (x - mu) / (sigma + CONFIG['eps'])
    return output * a + b
