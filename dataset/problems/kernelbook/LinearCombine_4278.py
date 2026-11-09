"""LinearCombine: softmax(w, axis=0) -> broadcast multiply with seq -> sum over axis 0."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearCombine_4278',
    'layers_num': 1,
    'seq_len': 8,
    'd1': 64,
    'd2': 256,
    'd3': 512,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    seq = jax.random.uniform(
        key, (CONFIG['seq_len'], CONFIG['d1'], CONFIG['d2'], CONFIG['d3']), dtype)
    # Matches the PyTorch init: torch.full((layers_num, 1, 1, 1), 1 / layers_num).
    w = jnp.full((CONFIG['layers_num'], 1, 1, 1), 1.0 / CONFIG['layers_num'], dtype)
    return seq, w


def workload(seq, w):
    nw = jax.nn.softmax(w, axis=0)
    return jnp.sum(seq * nw, axis=0)
