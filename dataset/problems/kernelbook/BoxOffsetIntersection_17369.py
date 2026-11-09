"""[bf16 variant] BoxOffsetIntersection: relu(Linear) -> mean(dim0) -> sigmoid(Linear) gate times min(dim0) of embeddings."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BoxOffsetIntersection_17369',
    'num_sets': 64,
    'batch': 1024,
    'dim': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    n, b, d = CONFIG['num_sets'], CONFIG['batch'], CONFIG['dim']
    embeddings = jax.random.uniform(k_x, (n, b, d), dtype)
    w1 = jax.random.normal(k_w1, (d, d), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (d,), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (d, d), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (d,), dtype) * 0.02
    return embeddings, w1, b1, w2, b2


def workload(embeddings, w1, b1, w2, b2):
    layer1_act = jax.nn.relu(embeddings @ w1 + b1)
    layer1_mean = jnp.mean(layer1_act, axis=0)
    gate = jax.nn.sigmoid(layer1_mean @ w2 + b2)
    offset = jnp.min(embeddings, axis=0)
    return offset * gate
