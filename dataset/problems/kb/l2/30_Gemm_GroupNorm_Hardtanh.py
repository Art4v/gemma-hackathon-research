"""GEMM (Linear) -> GroupNorm -> Hardtanh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '30_Gemm_GroupNorm_Hardtanh',
    'batch_size': 1024,
    'in_features': 8192,
    'out_features': 8192,
    'num_groups': 16,
    'eps': 1e-5,
    'hardtanh_min': -2.0,
    'hardtanh_max': 2.0,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    B = CONFIG['batch_size']
    K = CONFIG['in_features']
    N = CONFIG['out_features']
    x = jax.random.uniform(k_x, (B, K), dtype)
    w = jax.random.normal(k_w, (K, N), dtype) * 0.02  # (in, out) layout, x @ w
    b = jax.random.normal(k_b, (N,), dtype) * 0.02
    gamma = jnp.ones((N,), dtype)
    beta = jnp.zeros((N,), dtype)
    return (x, w, b, gamma, beta)


def workload(x, w, b, gamma, beta):
    B = x.shape[0]
    N = w.shape[1]
    G = CONFIG['num_groups']
    eps = CONFIG['eps']
    y = x @ w + b
    yg = y.reshape(B, G, N // G)
    mean = jnp.mean(yg, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=-1, keepdims=True)
    yn = (yg - mean) * jax.lax.rsqrt(var + eps)
    yn = yn.reshape(B, N) * gamma + beta
    return jnp.clip(yn, CONFIG['hardtanh_min'], CONFIG['hardtanh_max'])
