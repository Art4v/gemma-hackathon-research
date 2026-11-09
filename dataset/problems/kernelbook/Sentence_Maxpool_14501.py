"""[bf16 variant] Sentence_Maxpool: Linear -> max over dim 1 -> ReLU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Sentence_Maxpool_14501',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'word_dimension': 512,
    'output_dim': 512,
    'relu': True,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['word_dimension']), dtype)
    w = jax.random.normal(kw, (CONFIG['word_dimension'], CONFIG['output_dim']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['output_dim'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    y = jnp.matmul(x, w) + b
    y = jnp.max(y, axis=1)
    if CONFIG['relu']:
        y = jax.nn.relu(y)
    return y
