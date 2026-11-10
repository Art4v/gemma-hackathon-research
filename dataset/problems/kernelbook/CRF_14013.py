"""[bf16 variant] CRF potentials: Linear(hidden->tagset) then broadcast-add of a (tagset, tagset) transition matrix."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CRF_14013',
    'batch': 8,
    'seq_len': 1024,
    'hidden_dim': 2048,
    'tagset_size': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb, kt = jax.random.split(jax.random.key(0), 4)
    h, t = CONFIG['hidden_dim'], CONFIG['tagset_size']
    feats = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['seq_len'], h), dtype)
    w = jax.random.normal(kw, (h, t), dtype) * 0.02
    b = jax.random.normal(kb, (t,), dtype) * 0.02
    transitions = jax.random.normal(kt, (t, t), dtype) * 0.02
    return feats, w, b, transitions


def workload(feats, w, b, transitions):
    h, t = w.shape
    scores = feats.reshape(-1, h) @ w + b
    # out[n, from, to] = scores[n, to] + transitions[from, to]
    return scores[:, None, :] + transitions.reshape(1, t, t)
