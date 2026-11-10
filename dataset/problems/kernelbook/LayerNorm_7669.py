"""Per-sample LayerNorm: (x - mean) / (unbiased_std + eps) over all non-batch dims, then per-channel affine."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_7669',
    'batch': 16,
    'num_features': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['num_features'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    gamma = jnp.ones((CONFIG['num_features'],), dtype)
    beta = jnp.zeros((CONFIG['num_features'],), dtype)
    return x, gamma, beta


def workload(x, gamma, beta):
    n = x.shape[0]
    flat = x.reshape(n, -1)
    bshape = (n,) + (1,) * (x.ndim - 1)
    mean = jnp.mean(flat, axis=1).reshape(bshape)
    # torch .std() is unbiased (ddof=1); eps is added to std, not variance.
    std = jnp.std(flat, axis=1, ddof=1).reshape(bshape)
    y = (x - mean) / (std + CONFIG['eps'])
    pshape = (1, -1) + (1,) * (x.ndim - 2)
    return y * gamma.reshape(pshape) + beta.reshape(pshape)
