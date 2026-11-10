"""Linear with bias -> ReLU -> custom LayerNorm (last axis, ddof=1 std, eps=1e-3, scale/shift, identity if axis 1 is singleton)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearEncoder_2787',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'd_model': 256,
    'd_inner': 512,
    'eps': 0.001,
    'ddof': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    d_model = CONFIG['d_model']
    d_inner = CONFIG['d_inner']
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], d_model), dtype)
    w = jax.random.normal(k_w, (d_model, d_inner), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (d_inner,), dtype) * CONFIG['weight_scale']
    a_2 = jnp.ones((d_inner,), dtype)
    b_2 = jnp.zeros((d_inner,), dtype)
    return x, w, b, a_2, b_2


def workload(x, w, b, a_2, b_2):
    z = jnp.maximum(x @ w + b, 0)
    if z.shape[1] == 1:
        return z
    mu = jnp.mean(z, axis=-1, keepdims=True)
    sigma = jnp.std(z, axis=-1, keepdims=True, ddof=CONFIG['ddof'])
    return (z - mu) / (sigma + CONFIG['eps']) * a_2 + b_2
