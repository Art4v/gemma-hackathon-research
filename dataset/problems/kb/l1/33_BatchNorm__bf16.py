"""[bf16 variant] BatchNorm2d operation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '33_BatchNorm',
    'batch_size': 16,
    'features': 128,
    'dim1': 256,
    'dim2': 256,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['features'], CONFIG['dim1'], CONFIG['dim2']),
        dtype=dtype,
    )
    weight = jnp.ones((CONFIG['features'],), dtype=dtype)
    bias = jnp.zeros((CONFIG['features'],), dtype=dtype)
    return x, weight, bias

def workload(x, weight, bias):
    mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    var = jnp.var(x, axis=(0, 2, 3), keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + CONFIG['eps'])
    return x_norm * weight[None, :, None, None] + bias[None, :, None, None]
