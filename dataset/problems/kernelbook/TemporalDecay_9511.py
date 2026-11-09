"""TemporalDecay: gamma = exp(-relu(d @ W + b))."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'TemporalDecay_9511',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'input_size': 512,
    'rnn_hid_size': 512,
}


def create_inputs(dtype=jnp.float32):
    k_d, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    d = jax.random.uniform(
        k_d,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['input_size']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['input_size'], CONFIG['rnn_hid_size']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['rnn_hid_size'],), dtype) * 0.02
    return d, w, b


def workload(d, w, b):
    z = jnp.matmul(d, w, precision=lax.Precision.HIGHEST) + b
    return jnp.exp(-jax.nn.relu(z))
