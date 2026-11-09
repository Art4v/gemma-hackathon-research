"""[m1 M3_ragged variant of kb/l2/33_Gemm_Scale_BatchNorm] GEMM (Linear) -> per-feature scale -> BatchNorm1d with batch statistics, with out_features 8000 instead of 8192 so the feature axis is not a multiple of 128."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '33_Gemm_Scale_BatchNorm_m1_ragged',
    'batch_size': 1024,
    'in_features': 8192,
    'out_features': 8000,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch_size']
    K = CONFIG['in_features']
    N = CONFIG['out_features']
    kx, kw, kb, ks = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(kx, (B, K), dtype)
    w = jax.random.normal(kw, (K, N), dtype) * 0.02   # nn.Linear weight, stored (in, out)
    b = jax.random.normal(kb, (N,), dtype) * 0.02     # nn.Linear bias
    scale = jax.random.normal(ks, (N,), dtype)        # torch.randn(scale_shape)
    gamma = jnp.ones((N,), dtype)                     # BatchNorm weight
    beta = jnp.zeros((N,), dtype)                     # BatchNorm bias
    return (x, w, b, scale, gamma, beta)


def workload(x, w, b, scale, gamma, beta):
    eps = CONFIG['eps']
    y = jnp.dot(x, w, preferred_element_type=jnp.float32) + b
    y = y * scale
    mean = jnp.mean(y, axis=0, keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=0, keepdims=True)
    y = (y - mean) * jax.lax.rsqrt(var + eps)
    return (y * gamma + beta).astype(x.dtype)
