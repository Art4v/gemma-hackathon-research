"""AttPool: Linear(F->1) -> softmax over dim 4 -> weighted sum over dim 4 (keepdim)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttPool_18021',
    'shape': (8, 16, 32, 64, 256),
    'input_size': 256,
    'dim': 4,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, CONFIG['shape'], dtype)
    w = jax.random.normal(kw, (CONFIG['input_size'], 1), dtype) * 0.02
    b = jax.random.normal(kb, (1,), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    dim = CONFIG['dim']
    # ponytail: lin(x) has size 1 on `dim`, so softmax is all ones; kept literal to mirror PyTorch.
    scores = jax.nn.softmax(x @ w + b, axis=dim)
    return jnp.sum(scores * x, axis=dim, keepdims=True)
