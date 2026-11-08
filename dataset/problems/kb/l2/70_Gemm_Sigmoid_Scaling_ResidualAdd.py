"""GEMM (Linear) -> sigmoid -> scale -> residual add with the GEMM output."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '70_Gemm_Sigmoid_Scaling_ResidualAdd',
    'batch_size': 1024,
    'input_size': 8192,
    'hidden_size': 8192,
    'scaling_factor': 2.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch_size'], CONFIG['input_size']), dtype)
    w = jax.random.normal(
        kw, (CONFIG['input_size'], CONFIG['hidden_size']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['hidden_size'],), dtype) * 0.02
    return (x, w, b)


def workload(x, w, b):
    h = x @ w + b
    s = jax.nn.sigmoid(h)
    return s * CONFIG['scaling_factor'] + h
