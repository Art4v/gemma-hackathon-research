"""[bf16 variant] L2Normalization: sum(x*x, dim=1) -> sqrt -> (x.T / (div + eps)).T (full-axis-reversal transpose, as in PyTorch)."""
import jax
import jax.numpy as jnp

# ponytail: the PyTorch module's .T broadcasting only works when B == C == H == W, so all dims share N.
CONFIG = {
    'name': 'L2Normalization_17502',
    'batch': 64,
    'channels': 64,
    'height': 64,
    'width': 64,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    div = jnp.sqrt(jnp.sum(x * x, axis=1))
    # x.T reverses all axes and div is not transposed, so this matches the PyTorch broadcasting exactly.
    return (x.T / (div + CONFIG['eps'])).T
