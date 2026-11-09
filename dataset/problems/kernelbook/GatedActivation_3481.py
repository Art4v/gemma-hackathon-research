"""GatedActivation: tanh(x[:, :C/2]) * sigmoid(x[:, C/2:]) over NCHW channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GatedActivation_3481',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    c = x.shape[1]
    assert c % 2 == 0, 'x must have an even number of channels.'
    half = c // 2
    f, gate = x[:, :half], x[:, half:]
    return jnp.tanh(f) * jax.nn.sigmoid(gate)
