"""Relu_Adpt: capsule view -> L2 length over D -> relu(length - theta) * caps / (length + eps) -> flatten channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Relu_Adpt_5968',
    'batch': 8,
    'dim1': 4,
    'height': 512,
    'width': 512,
    'dim4': 4,
    'num_C': 4,
    'num_D': 4,
    'eps': 1e-4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['height'], CONFIG['width'], CONFIG['dim4']),
        dtype,
    )
    theta = jnp.zeros((1, CONFIG['num_C'], 1, 1, 1), dtype)
    return x, theta


def workload(x, theta):
    b, h, w = x.shape[0], x.shape[2], x.shape[3]
    num_c = theta.shape[1]
    caps = x.reshape(b, num_c, -1, h, w)
    length = jnp.sqrt(jnp.sum(caps * caps, axis=2, keepdims=True))
    out = jax.nn.relu(length - theta) * caps / (length + CONFIG['eps'])
    return out.reshape(b, -1, h, w)
