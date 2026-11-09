"""[bf16 variant] Net_6972: 5-layer MLP, Linear(784,512)->ReLU->Linear(512,256)->ReLU->Linear(256,128)->ReLU->Linear(128,64)->ReLU->Linear(64,10)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Net_6972',
    'batch': 32768,
    'layer_dims': (784, 512, 256, 128, 64, 10),
}


def create_inputs(dtype=jnp.bfloat16):
    dims = CONFIG['layer_dims']
    n_layers = len(dims) - 1
    keys = jax.random.split(jax.random.key(0), 1 + 2 * n_layers)
    x = jax.random.uniform(keys[0], (CONFIG['batch'], dims[0]), dtype=dtype)
    params = []
    for i in range(n_layers):
        w = jax.random.normal(keys[1 + 2 * i], (dims[i], dims[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (dims[i + 1],), dtype) * 0.02
        params.extend([w, b])
    return (x, *params)


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
    h = x.reshape(-1, w1.shape[0])
    h = jax.nn.relu(h @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    h = jax.nn.relu(h @ w3 + b3)
    h = jax.nn.relu(h @ w4 + b4)
    return h @ w5 + b5
