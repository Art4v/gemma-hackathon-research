"""[bf16 variant] Discriminator: flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax(dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Discriminator_1645',
    'batch': 8192,
    'n_layersDecod': 4,
    'hidden_size': 1024,
    'output_size': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    n = CONFIG['n_layersDecod']
    h = CONFIG['hidden_size']
    o = CONFIG['output_size']
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(keys[0], (b, n, h), dtype)
    w1 = jax.random.normal(keys[1], (n * h, h), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (h,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (h, h), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (h,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (h, o), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (o,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    x = x.reshape(-1, w1.shape[0])
    x = jax.nn.relu(x @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    # nn.Softmax() with no dim on a 2D tensor uses dim=1.
    return jax.nn.softmax(x @ w3 + b3, axis=-1)
