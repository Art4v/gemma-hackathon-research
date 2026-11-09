"""[bf16 variant] TransformerLinearXMCHead: drop padding row of label weight/bias embeddings, repeat per device."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TransformerLinearXMCHead_3300',
    'hidden_size': 1024,
    'num_labels': 65536,
    'num_device': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    kw, kb = jax.random.split(jax.random.key(0))
    num_labels = CONFIG['num_labels']
    hidden = CONFIG['hidden_size']
    # Last row is the padding_idx row, kept zero like the PyTorch init.
    W = jnp.concatenate([
        jax.random.normal(kw, (num_labels, hidden), dtype) * 0.02,
        jnp.zeros((1, hidden), dtype),
    ], axis=0)
    b = jnp.concatenate([
        jax.random.normal(kb, (num_labels, 1), dtype) * 0.02,
        jnp.zeros((1, 1), dtype),
    ], axis=0)
    return W, b


def workload(W, b):
    num_device = CONFIG['num_device']
    W_act = jnp.tile(W[:-1][None], (num_device, 1, 1))
    b_act = jnp.tile(b[:-1][None], (num_device, 1, 1))
    return W_act, b_act
