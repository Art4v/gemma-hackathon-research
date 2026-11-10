"""[bf16 variant] LIN: instance-norm and layer-norm blended by rho, then per-channel affine (gamma, beta)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LIN_3874',
    'batch': 16,
    'num_features': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    b, c = CONFIG['batch'], CONFIG['num_features']
    h, w = CONFIG['height'], CONFIG['width']
    x = jax.random.uniform(jax.random.key(0), (b, c, h, w), dtype)
    rho = jnp.zeros((1, c, 1, 1), dtype)
    gamma = jnp.ones((1, c, 1, 1), dtype)
    beta = jnp.zeros((1, c, 1, 1), dtype)
    return x, rho, gamma, beta


def workload(x, rho, gamma, beta):
    eps = CONFIG['eps']
    # torch.var defaults to unbiased (correction=1)
    in_mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    in_var = jnp.var(x, axis=(2, 3), keepdims=True, ddof=1)
    out_in = (x - in_mean) / jnp.sqrt(in_var + eps)
    ln_mean = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
    ln_var = jnp.var(x, axis=(1, 2, 3), keepdims=True, ddof=1)
    out_ln = (x - ln_mean) / jnp.sqrt(ln_var + eps)
    out = rho * out_in + (1 - rho) * out_ln
    return out * gamma + beta
