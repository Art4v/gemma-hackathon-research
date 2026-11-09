"""[bf16 variant] GEMM (Linear) -> Swish -> divide -> clamp -> tanh -> clamp."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '81_Gemm_Swish_Divide_Clamp_Tanh_Clamp',
    'batch_size': 1024,
    'in_features': 8192,
    'out_features': 8192,
    'divisor': 2.0,
    'clamp_min': -1.0,
    'clamp_max': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch_size'], CONFIG['in_features']), dtype)
    # nn.Linear weight stored (out, in) in PyTorch; passed here as (in, out).
    w = jax.random.normal(
        kw, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_features'],), dtype) * 0.02
    return (x, w, b)


def workload(x, w, b):
    y = x @ w + b
    y = y * jax.nn.sigmoid(y)
    y = y / CONFIG['divisor']
    y = jnp.clip(y, CONFIG['clamp_min'], CONFIG['clamp_max'])
    y = jnp.tanh(y)
    y = jnp.clip(y, CONFIG['clamp_min'], CONFIG['clamp_max'])
    return y
