"""[bf16 variant] Attention_17195: concat(curr_node, x) -> Linear(2H,1) -> softmax over neighbors -> weighted sum -> mean over nodes."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_17195',
    'batch': 8,
    'num_nodes': 128,
    'num_neighbors': 256,
    'n_h': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    shape = (CONFIG['batch'], CONFIG['num_nodes'], CONFIG['num_neighbors'], CONFIG['n_h'])
    x = jax.random.uniform(kx, shape, dtype)
    w = jax.random.normal(kw, (2 * CONFIG['n_h'], 1), dtype) * 0.02
    b = jax.random.normal(kb, (1,), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    curr_node = jnp.broadcast_to(x[:, :, 0:1, :], x.shape)
    stacked = jnp.concatenate((curr_node, x), axis=3)
    logits = (stacked @ w + b)[..., 0]
    weights = jax.nn.softmax(logits, axis=2)[..., None]
    x2 = jnp.sum(x * weights, axis=2)
    return jnp.mean(x2, axis=1)
