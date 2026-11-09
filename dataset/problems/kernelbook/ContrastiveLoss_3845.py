"""ContrastiveLoss: pairwise L2 distance (keepdim) -> margin-clamped contrastive loss -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ContrastiveLoss_3845',
    'batch': 16,
    'channels': 64,
    'height': 128,
    'width': 128,
    'margin': 2.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    output1 = jax.random.uniform(k1, shape, dtype)
    output2 = jax.random.uniform(k2, shape, dtype)
    label = jax.random.uniform(k3, shape, dtype)
    return output1, output2, label


def workload(output1, output2, label):
    # Matches torch pairwise_distance: eps is added to the difference before the norm.
    diff = output1 - output2 + CONFIG['eps']
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1, keepdims=True))
    # (B,C,H,1) distance broadcasts against the full (B,C,H,W) label, as in PyTorch.
    hinge = jnp.maximum(CONFIG['margin'] - dist, 0.0)
    loss = (1.0 - label) * dist ** 2 + label * hinge ** 2
    return jnp.mean(loss)
