"""[bf16 variant] Contrastive loss: pairwise_distance (L2 over last axis, eps=1e-6) -> label-weighted squared distance + squared hinge on margin -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ConLoss_2157',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'label_leading': 128,
    'margin': 2.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    output_shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    # diff drops the last axis and aligns to the trailing axes of label under broadcasting.
    label_shape = (CONFIG['label_leading'], CONFIG['batch'], CONFIG['channels'], CONFIG['height'])
    output1 = jax.random.uniform(k1, output_shape, dtype)
    output2 = jax.random.uniform(k2, output_shape, dtype)
    label = jax.random.uniform(k3, label_shape, dtype)
    return output1, output2, label


def workload(output1, output2, label):
    delta = output1 - output2 + CONFIG['eps']
    diff = jnp.sqrt(jnp.sum(jnp.square(delta), axis=-1))
    hinge = jnp.maximum(CONFIG['margin'] - diff, 0.0)
    loss = label * jnp.square(diff) + (1 - label) * jnp.square(hinge)
    return jnp.mean(loss)
