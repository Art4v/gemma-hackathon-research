"""BoxOffsetIntersection: relu(linear) -> mean over set axis -> sigmoid(linear) gate * min over set axis."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BoxOffsetIntersection_15460',
    'num_sets': 8,
    'batch': 32,
    'seq': 512,
    'dim': 512,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w1, k_w2 = jax.random.split(key, 3)
    d = CONFIG['dim']
    x = jax.random.uniform(
        k_x, (CONFIG['num_sets'], CONFIG['batch'], CONFIG['seq'], d), dtype)
    w1 = jax.random.normal(k_w1, (d, d), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (d, d), dtype) * 0.02
    b1 = jnp.zeros((d,), dtype)
    b2 = jnp.zeros((d,), dtype)
    return x, w1, w2, b1, b2


def workload(embeddings, w1, w2, b1, b2):
    hp = lax.Precision.HIGHEST
    layer1_act = jax.nn.relu(jnp.dot(embeddings, w1, precision=hp) + b1)
    layer1_mean = jnp.mean(layer1_act, axis=0)
    gate = jax.nn.sigmoid(jnp.dot(layer1_mean, w2, precision=hp) + b2)
    offset = jnp.min(embeddings, axis=0)
    return offset * gate
