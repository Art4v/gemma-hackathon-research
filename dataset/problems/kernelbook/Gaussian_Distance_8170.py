"""[bf16 variant] Gaussian_Distance: avgpool -> exp(logvar) -> pairwise Gaussian kernel sums -> vaa + vbb - 2*vab."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Gaussian_Distance_8170',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'kern': 1,
    'exponent_scale': -0.5,
    'cross_weight': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_mu_a, k_logvar_a, k_mu_b, k_logvar_b = jax.random.split(jax.random.key(0), 4)
    mu_a = jax.random.uniform(k_mu_a, shape, dtype)
    logvar_a = jax.random.uniform(k_logvar_a, shape, dtype)
    mu_b = jax.random.uniform(k_mu_b, shape, dtype)
    logvar_b = jax.random.uniform(k_logvar_b, shape, dtype)
    return mu_a, logvar_a, mu_b, logvar_b


def workload(mu_a, logvar_a, mu_b, logvar_b):
    kern = CONFIG['kern']
    area = kern * kern
    window = (1, 1, kern, kern)

    def avgpool(x):
        return lax.reduce_window(x, 0.0, lax.add, window, window, 'VALID') / area

    def kernel_sum(mu_x, var_x, mu_y, var_y):
        mu_x1 = mu_x.reshape(mu_x.shape[0], 1, -1)
        var_x1 = var_x.reshape(var_x.shape[0], 1, -1)
        mu_y2 = mu_y.reshape(1, mu_y.shape[0], -1)
        var_y2 = var_y.reshape(1, var_y.shape[0], -1)
        var_sum = var_x1 + var_y2
        exponent = jnp.square(mu_x1 - mu_y2) / var_sum * CONFIG['exponent_scale']
        return jnp.sum(jnp.exp(exponent) / jnp.sqrt(var_sum))

    mu_a = avgpool(mu_a)
    mu_b = avgpool(mu_b)
    var_a = avgpool(jnp.exp(logvar_a)) / area
    var_b = avgpool(jnp.exp(logvar_b)) / area

    vaa = kernel_sum(mu_a, var_a, mu_a, var_a)
    vab = kernel_sum(mu_a, var_a, mu_b, var_b)
    vbb = kernel_sum(mu_b, var_b, mu_b, var_b)
    return vaa + vbb - vab * CONFIG['cross_weight']
