"""[bf16 variant] Spatial attention (ASTGCN): (x@W1)@W2 @ (W3@x)^T -> +bs -> sigmoid -> Vs@ -> softmax(dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Spatial_Attention_layer_10594',
    'batch': 16,
    'num_of_vertices': 512,
    'in_channels': 64,
    'num_of_timesteps': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    n = CONFIG['num_of_vertices']
    f = CONFIG['in_channels']
    t = CONFIG['num_of_timesteps']
    kx, k1, k2, k3, kb, kv = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(kx, (b, n, f, t), dtype)
    w1 = jax.random.normal(k1, (t,), dtype) * 0.02
    w2 = jax.random.normal(k2, (f, t), dtype) * 0.02
    w3 = jax.random.normal(k3, (f,), dtype) * 0.02
    bs = jax.random.normal(kb, (1, n, n), dtype) * 0.02
    vs = jax.random.normal(kv, (n, n), dtype) * 0.02
    return x, w1, w2, w3, bs, vs


def workload(x, w1, w2, w3, bs, vs):
    lhs = jnp.matmul(jnp.matmul(x, w1), w2)            # (B, N, T)
    rhs = jnp.swapaxes(jnp.matmul(w3, x), -1, -2)      # (B, T, N)
    product = jnp.matmul(lhs, rhs)                     # (B, N, N)
    s = jnp.matmul(vs, jax.nn.sigmoid(product + bs))   # (B, N, N)
    return jax.nn.softmax(s, axis=1)
