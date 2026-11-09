"""LNN layer: abs -> +eps -> log1p -> matmul(W, .) -> expm1 -> relu -> reshape."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LNN_2986',
    'batch': 32,
    'extra_dim': 128,
    'num_fields': 128,
    'embed_dim': 128,
    'lnn_dim': 128,
    'eps': 1e-7,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['extra_dim'], CONFIG['num_fields'], CONFIG['embed_dim']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['lnn_dim'], CONFIG['num_fields']), dtype) * 0.02
    return x, w


def workload(x, w):
    embed_log = jnp.log1p(jnp.abs(x) + CONFIG['eps'])
    lnn_out = jnp.matmul(w, embed_log)  # (L, F) @ (..., F, E) -> (..., L, E)
    out = jax.nn.relu(jnp.expm1(lnn_out))
    return out.reshape(-1, w.shape[0] * x.shape[-1])
