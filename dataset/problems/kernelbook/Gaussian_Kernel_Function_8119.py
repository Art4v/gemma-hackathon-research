"""Gaussian kernel MMD-style loss: mean(exp(-||a_i-a_j||^2/s)) + mean(bb) - 2*mean(ab)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Gaussian_Kernel_Function_8119',
    'n_a': 8192,
    'n_b': 8192,
    'dim': 1024,
    'std': 4.0,
}


def create_inputs(dtype=jnp.float32):
    ka, kb = jax.random.split(jax.random.key(0))
    fa = jax.random.uniform(ka, (CONFIG['n_a'], CONFIG['dim']), dtype)
    fb = jax.random.uniform(kb, (CONFIG['n_b'], CONFIG['dim']), dtype)
    return fa, fb


def _gauss_mean(x, y, sigma):
    # Expanded form of ||x_i - y_j||^2: the N x N x D difference tensor is too large to materialize.
    x2 = jnp.sum(x * x, axis=-1)[:, None]
    y2 = jnp.sum(y * y, axis=-1)[None, :]
    xy = jnp.dot(x, y.T, precision=lax.Precision.HIGHEST)
    d2 = jnp.maximum(x2 + y2 - 2.0 * xy, 0.0)
    return jnp.mean(jnp.exp(-d2 / sigma))


def workload(fa, fb):
    sigma = CONFIG['std'] ** 2
    a = fa.reshape(-1, fa.shape[1])
    b = fb.reshape(-1, fb.shape[1])
    vaa = _gauss_mean(a, a, sigma)
    vbb = _gauss_mean(b, b, sigma)
    vab = _gauss_mean(a, b, sigma)
    return vaa + vbb - 2.0 * vab
