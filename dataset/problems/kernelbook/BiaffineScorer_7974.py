"""[bf16 variant] BiaffineScorer: append ones to both inputs, then nn.Bilinear (x1a^T W_o x2a + b_o)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BiaffineScorer_7974',
    'batch': 8,
    'dim1': 64,
    'dim2': 128,
    'input1_size': 128,
    'input2_size': 128,
    'output_size': 64,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    lead = (c['batch'], c['dim1'], c['dim2'])
    input1 = jax.random.uniform(k1, lead + (c['input1_size'],), dtype)
    input2 = jax.random.uniform(k2, lead + (c['input2_size'],), dtype)
    # PyTorch zero-inits these; random values keep the benchmark non-trivial.
    w = jax.random.normal(
        k3, (c['output_size'], c['input1_size'] + 1, c['input2_size'] + 1), dtype) * 0.02
    b = jax.random.normal(k4, (c['output_size'],), dtype) * 0.02
    return input1, input2, w, b


def _append_ones(x):
    return jnp.concatenate([x, jnp.ones(x.shape[:-1] + (1,), x.dtype)], axis=-1)


def workload(input1, input2, w, b):
    x1 = _append_ones(input1)
    x2 = _append_ones(input2)
    # Contract x1 with W first: intermediate is N*out*(in2+1), smaller than the x1 (x) x2 outer product.
    t = jnp.einsum('...i,oij->...oj', x1, w)
    return jnp.einsum('...oj,...j->...o', t, x2) + b
