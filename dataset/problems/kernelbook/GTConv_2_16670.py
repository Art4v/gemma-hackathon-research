"""[bf16 variant] GTConv_2: permute HWC to CHW, softmax channel weights over axis 0, weighted sum over channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GTConv_2_16670',
    'height': 1024,
    'width': 1024,
    'in_channels': 128,
    'out_channels': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_a, k_w = jax.random.split(key)
    A = jax.random.uniform(
        k_a,
        (CONFIG['height'], CONFIG['width'], CONFIG['in_channels']),
        dtype,
    )
    weight = jax.random.normal(k_w, (CONFIG['in_channels'], 1, 1), dtype) * 0.02
    return A, weight


def workload(A, weight):
    A = jnp.transpose(A, (2, 0, 1))
    return jnp.sum(jax.nn.softmax(weight, axis=0) * A, axis=0)
