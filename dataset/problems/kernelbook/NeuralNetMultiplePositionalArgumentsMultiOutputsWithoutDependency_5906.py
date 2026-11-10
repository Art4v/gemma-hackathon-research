"""Add -> two parallel (Linear -> Softmax(dim=1) -> ReLU) branches, two outputs."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NeuralNetMultiplePositionalArgumentsMultiOutputsWithoutDependency_5906',
    'batch': 8,
    'dim1': 64,
    'dim2': 128,
    'input_size': 512,
    'hidden_size': 512,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.key(0), 6)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['input_size'])
    d_in, d_h = CONFIG['input_size'], CONFIG['hidden_size']
    input1 = jax.random.uniform(k1, shape, dtype)
    input2 = jax.random.uniform(k2, shape, dtype)
    w1 = jax.random.normal(k3, (d_in, d_h), dtype) * 0.02
    b1 = jax.random.normal(k4, (d_h,), dtype) * 0.02
    w2 = jax.random.normal(k5, (d_in, d_h), dtype) * 0.02
    b2 = jax.random.normal(k6, (d_h,), dtype) * 0.02
    return input1, input2, w1, b1, w2, b2


def workload(input1, input2, w1, b1, w2, b2):
    x = input1 + input2
    axis = CONFIG['softmax_axis']
    out1 = jax.nn.relu(jax.nn.softmax(x @ w1 + b1, axis=axis))
    out2 = jax.nn.relu(jax.nn.softmax(x @ w2 + b2, axis=axis))
    return out1, out2
