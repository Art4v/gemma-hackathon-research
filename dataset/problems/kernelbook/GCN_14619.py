"""GCN unit: node-mixing 1x1 conv (+bias) -> residual add -> ReLU -> state-mixing 1x1 conv (no bias)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GCN_14619',
    'batch': 16,
    'num_state': 1024,
    'num_node': 1024,
}


def create_inputs(dtype=jnp.float32):
    b, s, n = CONFIG['batch'], CONFIG['num_state'], CONFIG['num_node']
    kx, k1, kb, k2 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(kx, (b, s, n), dtype)
    w1 = jax.random.normal(k1, (n, n), dtype) * 0.02  # (in_node, out_node)
    b1 = jax.random.normal(kb, (n,), dtype) * 0.02
    w2 = jax.random.normal(k2, (s, s), dtype) * 0.02  # (out_state, in_state)
    return x, w1, b1, w2


def workload(x, w1, b1, w2):
    h = jnp.matmul(x, w1) + b1 + x
    h = jax.nn.relu(h)
    return jnp.einsum('ts,bsm->btm', w2, h)
