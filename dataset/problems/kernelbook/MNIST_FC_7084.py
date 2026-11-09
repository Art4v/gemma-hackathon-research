"""[bf16 variant] MNIST_FC: view -> Linear(784,32) -> ReLU -> Linear(32,10) -> softmax(dim=1) -> view."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MNIST_FC_7084',
    'batch': 65536,
    'in_features': 784,
    'hidden': 32,
    'num_classes': 10,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    b, d, h, c = CONFIG['batch'], CONFIG['in_features'], CONFIG['hidden'], CONFIG['num_classes']
    x = jax.random.uniform(k_x, (b, d), dtype)
    w1 = jax.random.normal(k_w1, (d, h), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (h,), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (h, c), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (c,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    x = x.reshape(-1, CONFIG['in_features'])
    hdn = jax.nn.relu(x @ w1 + b1)
    out = jax.nn.softmax(hdn @ w2 + b2, axis=-1)
    return out.reshape(-1, CONFIG['num_classes'])
