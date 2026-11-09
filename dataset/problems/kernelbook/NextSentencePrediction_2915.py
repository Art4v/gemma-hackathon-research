"""NextSentencePrediction: Linear(H, 2) -> LogSoftmax(dim=-1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NextSentencePrediction_2915',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'hidden_size': 512,
    'num_classes': 2,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['hidden_size']), dtype)
    w = jax.random.normal(kw, (CONFIG['hidden_size'], CONFIG['num_classes']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['num_classes'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    scores = jnp.matmul(x, w) + b
    return jax.nn.log_softmax(scores, axis=-1)
