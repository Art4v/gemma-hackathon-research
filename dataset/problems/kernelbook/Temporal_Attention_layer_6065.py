"""Temporal attention: (x^T U1) U2 @ (U3 x) -> +be -> sigmoid -> Ve @ . -> softmax(dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Temporal_Attention_layer_6065',
    'batch': 16,
    'num_of_vertices': 512,
    'in_channels': 64,
    'num_of_timesteps': 128,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    n = CONFIG['num_of_vertices']
    f = CONFIG['in_channels']
    t = CONFIG['num_of_timesteps']
    kx, k1, k2, k3, kb, kv = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(kx, (b, n, f, t), dtype)
    u1 = jax.random.normal(k1, (n,), dtype) * 0.02
    u2 = jax.random.normal(k2, (f, n), dtype) * 0.02
    u3 = jax.random.normal(k3, (f,), dtype) * 0.02
    be = jax.random.normal(kb, (1, t, t), dtype) * 0.02
    ve = jax.random.normal(kv, (t, t), dtype) * 0.02
    return x, u1, u2, u3, be, ve


def workload(x, u1, u2, u3, be, ve):
    # x: (B, N, F, T)
    lhs = jnp.einsum('bnft,n->btf', x, u1) @ u2          # (B, T, N)
    rhs = jnp.einsum('f,bnft->bnt', u3, x)               # (B, N, T)
    product = jnp.matmul(lhs, rhs)                       # (B, T, T)
    e = jnp.matmul(ve, jax.nn.sigmoid(product + be))     # (B, T, T)
    return jax.nn.softmax(e, axis=1)
