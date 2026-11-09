"""VirtualBatchNorm (reference pass): per-element batch mean/unbiased var over dim 0, normalize, per-channel affine."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'VirtualBatchNorm_15446',
    'batch': 32,
    'num_features': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['num_features'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    weight = jnp.ones((CONFIG['num_features'],), dtype)
    bias = jnp.zeros((CONFIG['num_features'],), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    c = x.shape[1]
    mean = jnp.mean(x, axis=0, keepdims=True)
    # torch.var defaults to unbiased (ddof=1).
    var = jnp.var(x, axis=0, keepdims=True, ddof=1)
    y = (x - mean) / jnp.sqrt(var + CONFIG['eps'])
    return y * weight.reshape(1, c, 1, 1) + bias.reshape(1, c, 1, 1)
