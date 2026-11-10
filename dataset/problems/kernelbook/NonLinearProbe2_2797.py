"""NonLinearProbe2: Linear -> ReLU -> Linear over the last axis of a 4D tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NonLinearProbe2_2797',
    'batch': 8,
    'height': 128,
    'width': 128,
    'input_dim': 512,
    'num_hidden': 256,
    'num_classes': 256,
    'init_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(key, 5)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['input_dim']),
        dtype,
    )
    w1 = jax.random.normal(k_w1, (CONFIG['input_dim'], CONFIG['num_hidden']), dtype) * CONFIG['init_scale']
    b1 = jax.random.normal(k_b1, (CONFIG['num_hidden'],), dtype) * CONFIG['init_scale']
    w2 = jax.random.normal(k_w2, (CONFIG['num_hidden'], CONFIG['num_classes']), dtype) * CONFIG['init_scale']
    b2 = jax.random.normal(k_b2, (CONFIG['num_classes'],), dtype) * CONFIG['init_scale']
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2
