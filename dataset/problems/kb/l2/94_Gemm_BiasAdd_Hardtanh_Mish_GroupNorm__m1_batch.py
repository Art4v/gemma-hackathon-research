"""[m1 M1_batch variant of kb/l2/94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm] GEMM -> BiasAdd -> Hardtanh -> Mish -> GroupNorm applied to 8 sequences of 256 tokens each, with the linear and norm parameters shared."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm_m1_batch',
    'num_seqs': 8,
    'batch_size': 256,
    'in_features': 8192,
    'out_features': 8192,
    'num_groups': 256,
    'eps': 1e-5,
    'hardtanh_min': -1.0,
    'hardtanh_max': 1.0,
}


def create_inputs(dtype=jnp.float32):
    S = CONFIG['num_seqs']
    B = CONFIG['batch_size']
    K = CONFIG['in_features']
    N = CONFIG['out_features']
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(k0, (S, B, K), dtype)
    w = jax.random.normal(k1, (K, N), dtype) * 0.02   # nn.Linear weight, stored (in, out)
    b = jax.random.normal(k2, (N,), dtype) * 0.02     # nn.Linear bias
    bias = jax.random.normal(k3, (N,), dtype) * 0.02  # extra bias parameter
    gamma = jnp.ones((N,), dtype)                     # GroupNorm weight
    beta = jnp.zeros((N,), dtype)                     # GroupNorm bias
    return (x, w, b, bias, gamma, beta)


def _single(x, w, b, bias, gamma, beta):
    G = CONFIG['num_groups']
    eps = CONFIG['eps']
    B = x.shape[0]
    N = w.shape[1]

    y = x @ w + b
    y = y + bias
    y = jnp.clip(y, CONFIG['hardtanh_min'], CONFIG['hardtanh_max'])
    y = y * jnp.tanh(jax.nn.softplus(y))  # Mish

    # GroupNorm over (N, C): groups partition the channel dim.
    yg = y.reshape(B, G, N // G)
    mean = jnp.mean(yg, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=-1, keepdims=True)
    yg = (yg - mean) * jax.lax.rsqrt(var + eps)
    y = yg.reshape(B, N) * gamma + beta
    return y


def workload(x, w, b, bias, gamma, beta):
    return jax.vmap(_single, in_axes=(0, None, None, None, None, None))(x, w, b, bias, gamma, beta)
