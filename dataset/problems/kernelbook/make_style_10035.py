"""[bf16 variant] make_style: global avg pool over HW -> flatten -> L2-normalize over channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'make_style_10035',
    'batch': 16,
    'channels': 512,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    # Global average pool over the full spatial extent, flattened to (N, C).
    style = jnp.mean(x, axis=(2, 3))
    # No eps, matching the PyTorch module.
    norm = jnp.sum(style ** 2, axis=1, keepdims=True) ** 0.5
    return style / norm
