"""[bf16 variant] Net2_3500: Linear -> ReLU -> Sigmoid -> Linear -> Softmax(dim=1) on a 4D input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Net2_3500',
    'batch': 8,
    'dim1': 16,
    'dim2': 32,
    'shape_size': 16384,
    'mid_size': 128,  # round(sqrt(shape_size))
    'num_classes': 3,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k = jax.random.split(jax.random.key(0), 5)
    s, m, c = CONFIG['shape_size'], CONFIG['mid_size'], CONFIG['num_classes']
    x = jax.random.uniform(k[0], (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], s), dtype)
    w1 = jax.random.normal(k[1], (s, m), dtype) * 0.02
    b1 = jax.random.normal(k[2], (m,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (m, c), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.sigmoid(jax.nn.relu(x @ w1 + b1))
    y = h @ w2 + b2
    # Matches PyTorch Softmax(dim=1): axis 1 of the 4D tensor, not the class axis.
    return jax.nn.softmax(y, axis=CONFIG['softmax_axis'])
