"""[bf16 variant] Feed-forward: Linear -> ReLU -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FF_16887',
    'batch': 8,
    'dim1': 16,
    'dim2': 128,
    'input_size': 1024,
    'hidden_size': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    d, h = CONFIG['input_size'], CONFIG['hidden_size']
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], d), dtype)
    w1 = jax.random.normal(k1, (d, h), dtype) * 0.02
    b1 = jax.random.normal(kb1, (h,), dtype) * 0.02
    w2 = jax.random.normal(k2, (h, d), dtype) * 0.02
    b2 = jax.random.normal(kb2, (d,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    hidden = jax.nn.relu(jnp.matmul(x, w1, precision=jax.lax.Precision.HIGHEST) + b1)
    return jnp.matmul(hidden, w2, precision=jax.lax.Precision.HIGHEST) + b2
