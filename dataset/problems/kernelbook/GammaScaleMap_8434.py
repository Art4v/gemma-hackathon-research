"""Gamma scale map: x + eps -> Gamma shape k (closed form + Newton-Raphson over channels) -> mean_c(x) / k."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GammaScaleMap_8434',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-07,
    'k_eps': 1e-07,
    'newton_iters': 10,
    'reduce_dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def _trigamma(x):
    # Asymptotic series at x+1 plus the recurrence term 1/x^2, as in the torch module.
    z = x + 1.0
    zz = z * z
    a = 0.2 - 1.0 / (7.0 * zz)
    b = 1.0 - a / zz
    c = 1.0 + b / (3.0 * z)
    d = 1.0 + c / (2.0 * z)
    e = d / z
    return e + 1.0 / (x * x)


def _k_update(k, s):
    nm = jnp.log(k) - lax.digamma(k) - s
    dn = 1.0 / k - _trigamma(k)
    return k - nm / dn


def workload(x):
    dim = CONFIG['reduce_dim']
    x = x + CONFIG['eps']
    mean_x = jnp.mean(x, axis=dim)
    s = jnp.log(mean_x) - jnp.mean(jnp.log(x), axis=dim)
    s3 = s - 3.0
    rt = jnp.sqrt(s3 * s3 + 24.0 * s)
    k = (3.0 - s + rt) / (12.0 * s) + CONFIG['k_eps']
    for _ in range(CONFIG['newton_iters']):
        k = _k_update(k, s)
    return (1.0 / k) * mean_x
