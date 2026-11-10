"""[bf16 variant] Dice loss: sigmoid -> element-wise product -> global sums -> 1 - 2 * intersection / (union + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceLoss_7872',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-07,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key_output, key_target = jax.random.split(jax.random.key(0))
    output = jax.random.uniform(key_output, shape, dtype)
    target = jax.random.uniform(key_target, shape, dtype)
    return output, target


def workload(output, target):
    prediction = jax.nn.sigmoid(output)
    intersection = jnp.sum(prediction * target)
    union = jnp.sum(prediction) + jnp.sum(target) + CONFIG['eps']
    return 1 - 2 * intersection / union
