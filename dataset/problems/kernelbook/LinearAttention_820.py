"""[bf16 variant] LinearAttention: Linear(D->1) -> softmax over dim 1 -> alpha-weighted sum over dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearAttention_820',
    'batch': 8,
    'seq': 512,
    'heads': 128,
    'in_size': 128,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['seq'], CONFIG['heads'], CONFIG['in_size']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_size'], 1), dtype) * 0.02
    b = jax.random.normal(kb, (1,), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    axis = CONFIG['softmax_axis']
    alpha = jax.nn.softmax(x @ w + b, axis=axis)  # (B, S, A, 1)
    return jnp.sum(alpha * x, axis=axis)  # (B, A, D)
