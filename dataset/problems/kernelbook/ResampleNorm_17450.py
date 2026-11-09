"""[bf16 variant] ResampleNorm (no resample): x * sigmoid(mask) * 2 -> LayerNorm over last dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResampleNorm_17450',
    'batch': 16,
    'dim1': 64,
    'dim2': 128,
    'hidden': 512,
    'eps': 1e-5,
    'gate_scale': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['hidden'])
    x = jax.random.uniform(key, shape, dtype)
    mask = jnp.zeros((CONFIG['hidden'],), dtype)
    gamma = jnp.ones((CONFIG['hidden'],), dtype)
    beta = jnp.zeros((CONFIG['hidden'],), dtype)
    return x, mask, gamma, beta


def workload(x, mask, gamma, beta):
    x = x * jax.nn.sigmoid(mask) * CONFIG['gate_scale']
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta
