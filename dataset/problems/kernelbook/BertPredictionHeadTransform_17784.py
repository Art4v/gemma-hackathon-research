"""BertPredictionHeadTransform: Linear -> GELU (exact erf) -> LayerNorm."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BertPredictionHeadTransform_17784',
    'dim0': 8,
    'dim1': 8,
    'seq_len': 512,
    'hidden_size': 1024,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    h = CONFIG['hidden_size']
    x = jax.random.uniform(
        k_x, (CONFIG['dim0'], CONFIG['dim1'], CONFIG['seq_len'], h), dtype)
    w = jax.random.normal(k_w, (h, h), dtype) * 0.02
    b = jax.random.normal(k_b, (h,), dtype) * 0.02
    gamma = jnp.ones((h,), dtype)
    beta = jnp.zeros((h,), dtype)
    return x, w, b, gamma, beta


def workload(x, w, b, gamma, beta):
    y = x @ w + b
    y = jax.nn.gelu(y, approximate=False)
    mean = jnp.mean(y, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=-1, keepdims=True)
    y = (y - mean) * jax.lax.rsqrt(var + CONFIG['eps'])
    return y * gamma + beta
