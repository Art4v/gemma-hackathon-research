"""Gaussian_Distance: avgpool -> exp(logvar) -> pairwise Gaussian overlap sums (aa, ab, bb) -> vaa + vbb - 2*vab."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Gaussian_Distance_9230',
    'batch': 16,
    'channels': 64,
    'height': 64,
    'width': 64,
    'kern': 1,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    mu_a = jax.random.uniform(k1, shape, dtype)
    logvar_a = jax.random.uniform(k2, shape, dtype)
    mu_b = jax.random.uniform(k3, shape, dtype)
    logvar_b = jax.random.uniform(k4, shape, dtype)
    return mu_a, logvar_a, mu_b, logvar_b


def _avgpool(x):
    # NCHW AvgPool2d(kernel=stride=kern), no padding, floor mode (PyTorch defaults).
    k = CONFIG['kern']
    s = lax.reduce_window(x, jnp.array(0, x.dtype), lax.add,
                          (1, 1, k, k), (1, 1, k, k), 'VALID')
    return s / (k * k)


def _pair_sum(m1, v1, m2, v2):
    s = v1 + v2
    return jnp.sum(jnp.exp(-0.5 * (jnp.square(m1 - m2) / s + jnp.log(s))))


def workload(mu_a, logvar_a, mu_b, logvar_b):
    k = CONFIG['kern']
    n = mu_a.shape[0]
    mu_a = _avgpool(mu_a).reshape(n, -1)
    mu_b = _avgpool(mu_b).reshape(n, -1)
    var_a = (_avgpool(jnp.exp(logvar_a)) / (k * k)).reshape(n, -1)
    var_b = (_avgpool(jnp.exp(logvar_b)) / (k * k)).reshape(n, -1)
    vaa = _pair_sum(mu_a[:, None], var_a[:, None], mu_a[None], var_a[None])
    vab = _pair_sum(mu_a[:, None], var_a[:, None], mu_b[None], var_b[None])
    vbb = _pair_sum(mu_b[:, None], var_b[:, None], mu_b[None], var_b[None])
    return vaa + vbb - 2.0 * vab
