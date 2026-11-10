"""[bf16 variant] VarianceNorm2d: spatial variance (ddof=1) -> x / sqrt(var + eps) -> per-channel alpha scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'VarianceNorm2d_2852',
    'batch': 8,
    'num_features': 128,
    'height': 256,
    'width': 256,
    'eps': 1e-05,
    'ddof': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['num_features'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    alpha = jnp.ones((CONFIG['num_features'],), dtype)
    return x, alpha


def workload(x, alpha):
    variances = jnp.var(x, axis=(2, 3), ddof=CONFIG['ddof'], keepdims=True)
    h = x / jnp.sqrt(variances + CONFIG['eps'])
    return alpha.reshape((-1, alpha.shape[0], 1, 1)) * h
