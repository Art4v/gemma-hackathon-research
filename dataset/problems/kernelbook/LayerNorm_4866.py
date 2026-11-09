"""[bf16 variant] Custom LayerNorm: per-sample mean / unbiased std over (C,H,W), (x-mean)/(std+eps), per-channel affine."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_4866',
    'batch': 16,
    'num_features': 128,
    'height': 128,
    'width': 256,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
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
    mean = jnp.mean(flat, axis=1)
    # torch.Tensor.std defaults to the unbiased estimator
    std = jnp.std(flat, axis=1, ddof=1)
    bshape = (n,) + (1,) * (x.ndim - 1)
    y = (x - mean.reshape(bshape)) / (std.reshape(bshape) + CONFIG['eps'])
    cshape = (1, -1) + (1,) * (x.ndim - 2)
    return y * gamma.reshape(cshape) + beta.reshape(cshape)
