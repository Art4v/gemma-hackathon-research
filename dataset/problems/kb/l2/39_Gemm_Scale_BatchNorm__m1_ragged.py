"""[m1 M3_ragged variant of kb/l2/39_Gemm_Scale_BatchNorm] Linear (GEMM + bias) -> per-feature scale -> BatchNorm1d with batch statistics, over a ragged batch of 16300 rows instead of 16384 so the reduction axis is not a multiple of 128 or 8."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '39_Gemm_Scale_BatchNorm_m1_ragged',
    'batch_size': 16300,
    'in_features': 4096,
    'out_features': 4096,
    'scale_shape': (4096,),
    'eps': 1e-5,
    'momentum': 0.1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b, k_s = jax.random.split(jax.random.key(0), 4)
    B = CONFIG['batch_size']
    I = CONFIG['in_features']
    O = CONFIG['out_features']
    x = jax.random.uniform(k_x, (B, I), dtype)
    w = jax.random.normal(k_w, (I, O), dtype) * 0.02  # nn.Linear weight, stored (in, out)
    b = jax.random.normal(k_b, (O,), dtype) * 0.02
    scale = jax.random.normal(k_s, CONFIG['scale_shape'], dtype)  # mirrors torch.randn(scale_shape)
    gamma = jnp.ones((O,), dtype)
    beta = jnp.zeros((O,), dtype)
    return (x, w, b, scale, gamma, beta)


def workload(x, w, b, scale, gamma, beta):
    eps = CONFIG['eps']
    y = jnp.dot(x, w, preferred_element_type=jnp.float32) + b
    y = y * scale
    mean = jnp.mean(y, axis=0, keepdims=True)
    var = jnp.var(y, axis=0, keepdims=True)  # biased variance, as in PyTorch BN normalization
    y = (y - mean) * jax.lax.rsqrt(var + eps)
    y = y * gamma + beta
    return y.astype(x.dtype)
