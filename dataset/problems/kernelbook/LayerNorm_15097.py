"""[bf16 variant] LayerNorm_15097: mean-center, divide by (mean(x^2) + eps)**2, scale by gamma, shift by beta."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_15097',
    'batch': 8,
    'dim1': 64,
    'dim2': 128,
    'input_dim': 1024,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['input_dim'])
    x = jax.random.uniform(key, shape, dtype)
    gamma = jnp.ones((CONFIG['input_dim'],), dtype)
    beta = jnp.zeros((CONFIG['input_dim'],), dtype)
    return x, gamma, beta


def workload(x, gamma, beta):
    out = x - jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(out ** 2, axis=-1, keepdims=True)
    # Faithful to the source module: squares (var + eps) instead of taking sqrt.
    std = (variance + CONFIG['eps']) ** 2
    out = out / std
    return out * gamma + beta
