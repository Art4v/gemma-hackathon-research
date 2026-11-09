"""[bf16 variant] MulScalarNegative: identity quant stubs around x * -0.3."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MulScalarNegative_14063',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 128,
    'scalar': -0.3,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    # ponytail: QuantStub/DeQuantStub are identity in an unconverted float model.
    return x * jnp.asarray(CONFIG['scalar'], x.dtype)
