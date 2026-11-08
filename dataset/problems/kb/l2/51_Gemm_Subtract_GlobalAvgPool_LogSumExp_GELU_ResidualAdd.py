"""Gemm -> Subtract -> GlobalAvgPool -> LogSumExp -> GELU -> ResidualAdd."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd',
    'batch_size': 2048,
    'in_features': 8192,
    'out_features': 8192,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb, ks = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    fin = CONFIG['in_features']
    fout = CONFIG['out_features']
    x = jax.random.uniform(kx, (B, fin), dtype)
    w = jax.random.normal(kw, (fin, fout), dtype) * 0.02
    b = jax.random.normal(kb, (fout,), dtype) * 0.02
    sub = jax.random.normal(ks, (fout,), dtype) * 0.02
    return x, w, b, sub


def workload(x, w, b, sub):
    original_x = x
    y = x @ w + b
    y = y - sub
    y = jnp.mean(y, axis=1, keepdims=True)
    y = jax.nn.logsumexp(y, axis=1, keepdims=True)
    y = jax.nn.gelu(y, approximate=False)
    return y + original_x
