"""[bf16 variant] BPNet: (Linear -> ReLU -> Dropout[identity]) x 2 -> Linear -> ReLU on the last axis of a 4D tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BPNet_255',
    'batch': 8,
    'height': 128,
    'width': 128,
    'input_dim': 256,
    'level1': 512,
    'level2': 512,
    'output_dim': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['input_dim']),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (CONFIG['input_dim'], CONFIG['level1']), dtype) * scale
    b1 = jax.random.normal(keys[2], (CONFIG['level1'],), dtype) * scale
    w2 = jax.random.normal(keys[3], (CONFIG['level1'], CONFIG['level2']), dtype) * scale
    b2 = jax.random.normal(keys[4], (CONFIG['level2'],), dtype) * scale
    w3 = jax.random.normal(keys[5], (CONFIG['level2'], CONFIG['output_dim']), dtype) * scale
    b3 = jax.random.normal(keys[6], (CONFIG['output_dim'],), dtype) * scale
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jnp.maximum(x @ w1 + b1, 0)
    h = jnp.maximum(h @ w2 + b2, 0)
    return jnp.maximum(h @ w3 + b3, 0)
