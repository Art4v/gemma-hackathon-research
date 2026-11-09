"""[bf16 variant] Sentence_Maxpool: Linear -> ReLU -> max over dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Sentence_Maxpool_2203',
    'batch': 8,
    'seq_len': 128,
    'inner': 64,
    'word_dimension': 512,
    'output_dim': 512,
    'relu': True,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['inner'], CONFIG['word_dimension']), dtype)
    w = jax.random.normal(kw, (CONFIG['word_dimension'], CONFIG['output_dim']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['output_dim'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    y = jnp.matmul(x, w) + b
    if CONFIG['relu']:
        y = jax.nn.relu(y)
    return jnp.max(y, axis=1)
