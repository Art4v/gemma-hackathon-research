"""[bf16 variant] Generative_Model: Linear -> ReLU -> Linear -> ReLU -> Linear -> split/concat along axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Generative_Model_10069',
    'batch': 8,
    'dim1': 32,
    'dim2': 128,
    'input_size': 1024,
    'hidden_size_1': 1024,
    'hidden_size_2': 1024,
    'output_size': 1024,
    'n_classes': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(k[0], (c['batch'], c['dim1'], c['dim2'], c['input_size']), dtype)
    w1 = jax.random.normal(k[1], (c['input_size'], c['hidden_size_1']), dtype) * 0.02
    b1 = jax.random.normal(k[2], (c['hidden_size_1'],), dtype) * 0.02
    w2 = jax.random.normal(k[3], (c['hidden_size_1'], c['hidden_size_2']), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c['hidden_size_2'],), dtype) * 0.02
    w3 = jax.random.normal(k[5], (c['hidden_size_2'], c['output_size']), dtype) * 0.02
    b3 = jax.random.normal(k[6], (c['output_size'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    out = h @ w3 + b3
    n = CONFIG['n_classes']
    # Matches PyTorch: slices are on axis 1, and the concat rebuilds `out`.
    features = out[:, 0:-n]
    labels = out[:, -n:]
    return jnp.concatenate((features, labels), axis=1)
