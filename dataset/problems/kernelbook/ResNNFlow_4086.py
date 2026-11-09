"""[bf16 variant] ResNNFlow with an empty Sequential: sigmoid(gate) * x + (1 - sigmoid(gate)) * x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResNNFlow_4086',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'gate_size': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_g = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    gate = jax.random.normal(k_g, (CONFIG['gate_size'],), dtype) * 0.02
    return x, gate


def workload(x, gate):
    # The Sequential has no submodules, so the transformed branch equals the input.
    out = x
    g = jax.nn.sigmoid(gate)
    return g * out + (1 - g) * x
