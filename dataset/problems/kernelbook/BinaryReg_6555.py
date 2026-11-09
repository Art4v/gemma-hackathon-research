"""[bf16 variant] BinaryReg: sigmoid -> |p - 0.5| clamped at min_diff -> reciprocal -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BinaryReg_6555',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'center': 0.5,
    'min_diff': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    pred = jax.random.uniform(jax.random.key(0), shape, dtype)
    return (pred,)


def workload(pred):
    # ponytail: mask=None path only; KernelBench get_inputs never passes a mask.
    p = jax.nn.sigmoid(pred)
    diff = jnp.maximum(jnp.abs(p - CONFIG['center']), CONFIG['min_diff'])
    loss = 1.0 / diff
    return jnp.mean(loss)
