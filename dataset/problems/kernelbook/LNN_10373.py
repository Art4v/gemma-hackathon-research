"""LNN: abs -> +eps -> log1p -> matmul(weight, x) -> expm1 -> relu -> reshape."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LNN_10373',
    'batch': 64,
    'outer': 16,
    'num_fields': 128,
    'embed_dim': 128,
    'lnn_dim': 128,
    'eps': 1e-7,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['outer'], CONFIG['num_fields'], CONFIG['embed_dim']),
        dtype,
    )
    weight = jax.random.normal(k_w, (CONFIG['lnn_dim'], CONFIG['num_fields']), dtype) * 0.02
    return x, weight


def workload(x, weight):
    lnn_dim = weight.shape[0]
    embed_dim = x.shape[-1]
    x_log = jnp.log1p(jnp.abs(x) + CONFIG['eps'])
    # (L, F) @ (..., F, E) -> (..., L, E), same broadcasting as torch.matmul
    lnn_out = jnp.matmul(weight, x_log, precision=lax.Precision.HIGHEST)
    out = jax.nn.relu(jnp.expm1(lnn_out))
    return out.reshape(-1, lnn_dim * embed_dim)
