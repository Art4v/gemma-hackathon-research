"""[bf16 variant] Linear (no bias) -> MaxPool1d(k) over features -> Linear (no bias)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearMaxPoolLinearModel_4351',
    'batch': 16384,
    'in_features': 4096,
    'pool_kernel': 4,
    'out_features': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_w2 = jax.random.split(jax.random.key(0), 3)
    b, f, k, o = CONFIG['batch'], CONFIG['in_features'], CONFIG['pool_kernel'], CONFIG['out_features']
    x = jax.random.uniform(k_x, (b, f), dtype)
    w1 = jax.random.normal(k_w1, (f, f), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (f // k, o), dtype) * 0.02
    return x, w1, w2


def workload(x, w1, w2):
    k = CONFIG['pool_kernel']
    h = x @ w1
    b, f = h.shape
    n = f // k
    # MaxPool1d(k), stride k: drop the trailing remainder like PyTorch, then max per window.
    pooled = jnp.max(h[:, : n * k].reshape(b, n, k), axis=-1)
    return pooled @ w2
