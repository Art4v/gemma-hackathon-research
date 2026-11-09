"""[bf16 variant] Linear -> ReLU -> Linear -> Softmax(dim=1) on a 4D input (Net_12881)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Net_12881',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'input_size': 512,
    'hidden_size': 512,
    'output_size': 512,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kx, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (c['batch'], c['dim1'], c['dim2'], c['input_size']), dtype)
    w1 = jax.random.normal(k1, (c['input_size'], c['hidden_size']), dtype) * 0.02
    b1 = jnp.zeros((c['hidden_size'],), dtype)
    w2 = jax.random.normal(k2, (c['hidden_size'], c['output_size']), dtype) * 0.02
    b2 = jnp.zeros((c['output_size'],), dtype)
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    out = x.astype(jnp.float32) @ w1 + b1
    out = jax.nn.relu(out)
    out = out @ w2 + b2
    # Softmax over axis 1 of the 4D tensor, matching nn.Softmax(dim=1); not the feature axis.
    return jax.nn.softmax(out, axis=CONFIG['softmax_axis'])
