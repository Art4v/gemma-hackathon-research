"""ContrastiveLoss: pairwise_distance(last dim, eps) -> (1-y)*d^2 + y*relu(margin-d)^2 -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ContrastiveLoss_1446',
    # All four dims must be equal: the (N, C, H) distance is right-aligned against the 4D label.
    'n': 64,
    'c': 64,
    'h': 64,
    'w': 64,
    'margin': 2.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['n'], CONFIG['c'], CONFIG['h'], CONFIG['w'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    output1 = jax.random.uniform(k1, shape, dtype)
    output2 = jax.random.uniform(k2, shape, dtype)
    label = jax.random.uniform(k3, shape, dtype)
    return output1, output2, label


def workload(output1, output2, label):
    diff = output1 - output2 + CONFIG['eps']
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # (N, C, H), broadcasts against label like torch
    hinge = jnp.maximum(CONFIG['margin'] - dist, 0.0)
    return jnp.mean((1.0 - label) * dist ** 2 + label * hinge ** 2)
