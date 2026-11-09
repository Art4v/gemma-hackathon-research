"""[bf16 variant] ContrastiveLoss: pairwise_distance (last dim, keepdim) -> contrastive hinge loss -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ContrastiveLoss_4327',
    'batch': 16,
    'channels': 64,
    'height': 128,
    'width': 128,
    'margin': 5.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    output1 = jax.random.uniform(k1, shape, dtype)
    output2 = jax.random.uniform(k2, shape, dtype)
    label = jax.random.uniform(k3, shape, dtype)
    return output1, output2, label


def workload(output1, output2, label):
    # PyTorch pairwise_distance adds eps to the difference before the norm.
    diff = output1 - output2 + CONFIG['eps']
    d = jnp.sqrt(jnp.sum(diff * diff, axis=-1, keepdims=True))
    hinge = jnp.maximum(CONFIG['margin'] - d, 0.0)
    loss = (1.0 - label) * d * d + label * hinge * hinge
    return jnp.mean(loss)
