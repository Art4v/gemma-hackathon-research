"""GateLayer: concat -> Linear(2D->D) -> Linear(D->1) -> sigmoid gate -> gate*x1 + (1-gate)*x2."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GateLayer_1320',
    'batch': 16,
    'seq_len': 2048,
    'input_dim': 1024,
}


def create_inputs(dtype=jnp.float32):
    b, s, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['input_dim']
    k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.key(0), 6)
    input1 = jax.random.uniform(k1, (b, s, d), dtype)
    input2 = jax.random.uniform(k2, (b, s, d), dtype)
    w1 = jax.random.normal(k3, (2 * d, d), dtype) * 0.02
    b1 = jax.random.normal(k4, (d,), dtype) * 0.02
    w2 = jax.random.normal(k5, (d, 1), dtype) * 0.02
    b2 = jax.random.normal(k6, (1,), dtype) * 0.02
    return input1, input2, w1, b1, w2, b2


def workload(input1, input2, w1, b1, w2, b2):
    norm_input = jnp.concatenate([input1, input2], axis=-1) @ w1 + b1
    gate = jax.nn.sigmoid(norm_input @ w2 + b2)
    return gate * input1 + (1 - gate) * input2
