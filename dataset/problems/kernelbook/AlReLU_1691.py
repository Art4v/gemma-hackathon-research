"""AlReLU: max(|alpha * x|, x) elementwise."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AlReLU_1691',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'alpha': 0.01,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    # Normal instead of torch.rand's [0,1) so the negative branch is exercised.
    x = jax.random.normal(key, shape, dtype)
    return (x,)


def workload(x):
    alpha = jnp.asarray(CONFIG['alpha'], dtype=x.dtype)
    return jnp.maximum(jnp.abs(alpha * x), x)
