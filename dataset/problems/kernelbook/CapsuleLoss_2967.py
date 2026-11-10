"""[bf16 variant] CapsuleLoss: margin loss (relu-squared, label-weighted sums) + scaled sum-MSE reconstruction loss."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CapsuleLoss_2967',
    'batch': 8,
    'channels': 32,
    'height': 256,
    'width': 256,
    'upper': 0.9,
    'lower': 0.1,
    'lmda': 0.5,
    'recon_scale': 0.0005,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    images = jax.random.uniform(k1, shape, dtype)
    labels = jax.random.uniform(k2, shape, dtype)
    logits = jax.random.uniform(k3, shape, dtype)
    reconstructions = jax.random.uniform(k4, shape, dtype)
    return images, labels, logits, reconstructions


def workload(images, labels, logits, reconstructions):
    left = jnp.square(jax.nn.relu(CONFIG['upper'] - logits))
    right = jnp.square(jax.nn.relu(logits - CONFIG['lower']))
    margin_loss = jnp.sum(labels * left) + CONFIG['lmda'] * jnp.sum((1 - labels) * right)
    diff = jnp.reshape(reconstructions, images.shape) - images
    reconstruction_loss = jnp.sum(jnp.square(diff))
    return margin_loss + CONFIG['recon_scale'] * reconstruction_loss
