"""[bf16 variant] Norm_5: custom LayerNorm over last axis (mean, biased var, rsqrt, affine g/b)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Norm_5',
    'batch': 16,
    'dim1': 64,
    'dim2': 128,
    'n_state': 512,
    'axis': -1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['n_state']),
        dtype,
    )
    g = jnp.ones((CONFIG['n_state'],), dtype)
    b = jnp.zeros((CONFIG['n_state'],), dtype)
    return x, g, b


def workload(x, g, b):
    axis = CONFIG['axis']
    u = jnp.mean(x, axis=axis, keepdims=True)
    d = x - u
    s = jnp.mean(jnp.square(d), axis=axis, keepdims=True)
    y = d * jax.lax.rsqrt(s + CONFIG['eps'])
    return y * g + b
