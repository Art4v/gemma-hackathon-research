"""[bf16 variant] FFN_7153: concat -> Linear -> tanh-GELU -> Linear -> split -> residual add -> per-segment LayerNorm."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FFN_7153',
    'batch': 8,
    'low_num': 256,
    'mid_num': 128,
    'hig_num': 128,
    'extra': 16,
    'embedding_dim': 512,
    'hidden_unit': 2048,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 5)
    b, x, e, h = CONFIG['batch'], CONFIG['extra'], CONFIG['embedding_dim'], CONFIG['hidden_unit']
    low = jax.random.uniform(keys[0], (b, CONFIG['low_num'], x, e), dtype)
    mid = jax.random.uniform(keys[1], (b, CONFIG['mid_num'], x, e), dtype)
    hig = jax.random.uniform(keys[2], (b, CONFIG['hig_num'], x, e), dtype)
    w1 = jax.random.normal(keys[3], (e, h), dtype) * 0.02
    w2 = jax.random.normal(keys[4], (h, e), dtype) * 0.02
    ones = jnp.ones((e,), dtype)
    zeros = jnp.zeros((e,), dtype)
    return (low, mid, hig, w1, w2, ones, zeros, ones, zeros, ones, zeros)


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _layer_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta


def workload(low, mid, hig, w1, w2, low_g, low_b, mid_g, mid_b, hig_g, hig_b):
    n_low, n_mid = low.shape[1], mid.shape[1]
    cated = jnp.concatenate((low, mid, hig), axis=1)
    out = _gelu_tanh(cated @ w1) @ w2
    o_low = out[:, :n_low]
    o_mid = out[:, n_low:n_low + n_mid]
    o_hig = out[:, n_low + n_mid:]
    return (
        _layer_norm(low + o_low, low_g, low_b),
        _layer_norm(mid + o_mid, mid_g, mid_b),
        _layer_norm(hig + o_hig, hig_g, hig_b),
    )
