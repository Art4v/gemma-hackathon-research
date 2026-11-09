"""[bf16 variant] CenterIntersection: relu(linear) -> linear -> softmax(dim=0) -> weighted sum over dim 0."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CenterIntersection_15454',
    'shape': (8, 128, 128, 512),
    'dim': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    dim = CONFIG['dim']
    k_x, k_w1, k_w2, k_b1, k_b2 = jax.random.split(jax.random.key(0), 5)
    embeddings = jax.random.uniform(k_x, CONFIG['shape'], dtype)
    w1 = jax.random.normal(k_w1, (dim, dim), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (dim, dim), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (dim,), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (dim,), dtype) * 0.02
    return embeddings, w1, w2, b1, b2


def workload(embeddings, w1, w2, b1, b2):
    # w1, w2 are (in, out).
    h = jax.nn.relu(embeddings @ w1 + b1)
    attention = jax.nn.softmax(h @ w2 + b2, axis=0)
    return jnp.sum(attention * embeddings, axis=0)
