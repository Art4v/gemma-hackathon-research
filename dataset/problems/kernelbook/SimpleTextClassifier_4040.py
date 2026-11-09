"""[bf16 variant] SimpleTextClassifier: Linear -> clamp(min=0) -> Linear -> log_softmax(dim=1) on a 4-D input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SimpleTextClassifier_4040',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'vocab_size': 512,
    'hidden': 128,
    'num_labels': 128,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['dim1'], c['dim2'], c['vocab_size']), dtype)
    w1 = jax.random.normal(k1, (c['vocab_size'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['hidden'], c['num_labels']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['num_labels'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    hidden = jnp.maximum(x @ w1 + b1, 0)
    logits = hidden @ w2 + b2
    # PyTorch's log_softmax(dim=1) on a 4-D tensor normalizes over axis 1, not the label axis.
    return jax.nn.log_softmax(logits, axis=CONFIG['softmax_axis'])
