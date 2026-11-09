"""[bf16 variant] Net4: 6-layer MLP (Linear+ReLU x5, then Linear to 1) applied over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Net4_7283',
    'lead_shape': (16, 32, 128),
    'input_size': 512,
    'hidden_sizes': (400, 300, 200, 100, 60),
    'output_size': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    sizes = (CONFIG['input_size'],) + CONFIG['hidden_sizes'] + (CONFIG['output_size'],)
    n_layers = len(sizes) - 1
    keys = jax.random.split(jax.random.key(0), 1 + 2 * n_layers)
    x = jax.random.uniform(keys[0], CONFIG['lead_shape'] + (CONFIG['input_size'],), dtype)
    params = []
    for i in range(n_layers):
        w = jax.random.normal(keys[1 + 2 * i], (sizes[i], sizes[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (sizes[i + 1],), dtype) * 0.02
        params.extend([w, b])
    return (x, *params)


def workload(x, *params):
    weights = params[0::2]
    biases = params[1::2]
    out = x
    for i, (w, b) in enumerate(zip(weights, biases)):
        out = out @ w + b
        if i < len(weights) - 1:
            out = jax.nn.relu(out)
    return out
