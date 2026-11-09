"""DiffLoss: flatten -> batch-mean center -> row L2 normalize -> (x1^T @ x2)^2 mean."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DiffLoss_286',
    'batch': 32768,
    'channels': 4,
    'height': 16,
    'width': 16,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    input1 = jax.random.uniform(k1, shape, dtype)
    input2 = jax.random.uniform(k2, shape, dtype)
    return input1, input2


def _center_normalize(x):
    x = x.reshape(x.shape[0], -1)
    x = x - jnp.mean(x, axis=0, keepdims=True)
    norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))
    return x / (norm + CONFIG['eps'])


def workload(input1, input2):
    a = _center_normalize(input1)
    b = _center_normalize(input2)
    gram = jnp.matmul(a.T, b, precision=lax.Precision.HIGHEST)
    return jnp.mean(gram ** 2)
