"""[bf16 variant] HighwayCNN: relu(Linear(x)) * softmax_dim1(Linear_gate(x)) + (1 - gate) * x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'HighwayCNN_10675',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'input_size': 512,
    'gate_bias': -1.0,
    # PyTorch's softmax with no dim argument uses dim=1 for 4D input.
    'softmax_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kwn, kbn, kwg = jax.random.split(key, 4)
    d = CONFIG['input_size']
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], d), dtype)
    w_normal = jax.random.normal(kwn, (d, d), dtype) * 0.02
    b_normal = jax.random.normal(kbn, (d,), dtype) * 0.02
    w_gate = jax.random.normal(kwg, (d, d), dtype) * 0.02
    b_gate = jnp.full((d,), CONFIG['gate_bias'], dtype)
    return x, w_normal, b_normal, w_gate, b_gate


def workload(x, w_normal, b_normal, w_gate, b_gate):
    normal = jax.nn.relu(x @ w_normal + b_normal)
    gate = jax.nn.softmax(x @ w_gate + b_gate, axis=CONFIG['softmax_dim'])
    return normal * gate + (1.0 - gate) * x
