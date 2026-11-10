"""[bf16 variant] Triplet contrastive loss: pairwise L2 distance (eps-shifted) -> mean -> margin hinge squared -> label-weighted mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ContrastiveLoss_11284',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'margin': 0.2,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    output1 = jax.random.uniform(k1, shape, dtype)
    output2 = jax.random.uniform(k2, shape, dtype)
    output3 = jax.random.uniform(k3, shape, dtype)
    label = jax.random.uniform(k4, shape, dtype)
    return output1, output2, output3, label


def _pairwise_distance(x1, x2):
    diff = x1 - x2 + CONFIG['eps']
    return jnp.sqrt(jnp.sum(diff * diff, axis=-1, keepdims=True))


def workload(output1, output2, output3, label):
    margin = CONFIG['margin']
    d12 = jnp.mean(_pairwise_distance(output1, output2))
    d13 = jnp.mean(_pairwise_distance(output1, output3))
    term_a = jnp.maximum(margin + d13 - d12, 0.0) ** 2
    term_b = jnp.maximum(margin + d12 - d13, 0.0) ** 2
    return jnp.mean((1 - label) * term_a + label * term_b)
