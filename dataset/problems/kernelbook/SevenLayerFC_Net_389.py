"""SevenLayerFC_Net: five (Linear + ReLU) layers, then Linear and log_softmax over axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SevenLayerFC_Net_389',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'd_in': 512,
    'hidden': (2048, 1024, 512, 256, 128),
    'd_out': 128,
    'log_softmax_axis': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    sizes = (CONFIG['d_in'],) + CONFIG['hidden'] + (CONFIG['d_out'],)
    num_layers = len(sizes) - 1
    keys = jax.random.split(jax.random.key(0), 2 * num_layers + 1)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['d_in']),
        dtype,
    )
    params = []
    for i in range(num_layers):
        w = jax.random.normal(keys[2 * i + 1], (sizes[i], sizes[i + 1]), dtype) * CONFIG['weight_scale']
        b = jax.random.normal(keys[2 * i + 2], (sizes[i + 1],), dtype) * CONFIG['weight_scale']
        params.extend([w, b])
    return (x, *params)


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    h = jax.nn.relu(h @ w3 + b3)
    h = jax.nn.relu(h @ w4 + b4)
    h = jax.nn.relu(h @ w5 + b5)
    logits = h @ w6 + b6
    return jax.nn.log_softmax(logits, axis=CONFIG['log_softmax_axis'])
