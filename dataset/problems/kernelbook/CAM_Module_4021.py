"""CAM_Module_4021: channel attention via Gram matrix, max-minus-energy softmax, attention matmul, gamma-scaled residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CAM_Module_4021',
    'batch': 8,
    'channels': 256,
    'height': 128,
    'width': 128,
    'gamma_shape': (1,),
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    gamma = jnp.ones(CONFIG['gamma_shape'], dtype)
    return x, gamma


def workload(x, gamma):
    b, c, h, w = x.shape
    proj = x.reshape(b, c, -1)
    energy = proj @ jnp.swapaxes(proj, 1, 2)
    energy_new = jnp.max(energy, axis=-1, keepdims=True) - energy
    attention = jax.nn.softmax(energy_new, axis=-1)
    out = (attention @ proj).reshape(b, c, h, w)
    return gamma * out + x
