"""ILN: instance norm and layer norm (unbiased var) mixed by rho, then gamma/beta affine."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ILN_14443',
    'batch': 16,
    'num_features': 256,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['num_features'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(jax.random.key(0), shape, dtype)
    pshape = (1, CONFIG['num_features'], 1, 1)
    rho = jnp.zeros(pshape, dtype)
    gamma = jnp.ones(pshape, dtype)
    beta = jnp.zeros(pshape, dtype)
    return x, rho, gamma, beta


def _normalize(x, axes, eps):
    # torch.var defaults to unbiased (correction=1)
    mean = jnp.mean(x, axis=axes, keepdims=True)
    var = jnp.var(x, axis=axes, keepdims=True, ddof=1)
    return (x - mean) / jnp.sqrt(var + eps)


def workload(x, rho, gamma, beta):
    eps = CONFIG['eps']
    out_in = _normalize(x, (2, 3), eps)
    out_ln = _normalize(x, (1, 2, 3), eps)
    out = rho * out_in + (1 - rho) * out_ln
    return out * gamma + beta
