"""[bf16 variant] Linear (GEMM + bias) -> ReLU -> divide by constant."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '63_Gemm_ReLU_Divide',
    'batch_size': 1024,
    'in_features': 8192,
    'out_features': 8192,
    'divisor': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch_size'], CONFIG['in_features']), dtype)
    # nn.Linear weight is (out, in) in PyTorch; stored here as (in, out).
    w = jax.random.normal(
        kw, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_features'],), dtype) * 0.02
    return (x, w, b)


def workload(x, w, b):
    y = jnp.dot(x, w, preferred_element_type=jnp.float32) + b
    y = jnp.maximum(y, 0.0)
    y = y / CONFIG['divisor']
    return y.astype(x.dtype)
