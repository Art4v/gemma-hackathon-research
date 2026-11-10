"""Contrastive loss: pairwise_distance(x1, x2) -> margin-hinge contrastive mean."""
import jax
import jax.numpy as jnp

# ponytail: all four dims are equal because PyTorch broadcasts the (N,N,N) distance
# against the (N,N,N,N) label. 64 instead of 128 keeps inputs under 600 MB.
CONFIG = {
    'name': 'ContrastiveLoss_3514',
    'n0': 64,
    'n1': 64,
    'n2': 64,
    'n3': 64,
    'margin': 2.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['n0'], CONFIG['n1'], CONFIG['n2'], CONFIG['n3'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    output1 = jax.random.uniform(k1, shape, dtype)
    output2 = jax.random.uniform(k2, shape, dtype)
    label = jax.random.uniform(k3, shape, dtype)
    return output1, output2, label


def workload(output1, output2, label):
    diff = output1 - output2 + CONFIG['eps']
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    hinge = jnp.maximum(CONFIG['margin'] - dist, 0.0)
    loss = (1.0 - label) * dist ** 2 + label * hinge ** 2
    return jnp.mean(loss)
