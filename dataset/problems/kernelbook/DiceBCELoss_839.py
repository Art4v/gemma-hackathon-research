"""[bf16 variant] DiceBCELoss: sigmoid -> flatten -> dice loss + binary cross entropy (mean)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceBCELoss_839',
    'batch': 16,
    'channels': 4,
    'height': 512,
    'width': 512,
    'smooth': 1.0,
    'log_clamp': -100.0,  # PyTorch BCE clamps log terms to >= -100
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    inputs = jax.random.uniform(k1, shape, dtype)
    targets = jax.random.uniform(k2, shape, dtype)
    return inputs, targets


def workload(inputs, targets):
    smooth = CONFIG['smooth']
    clamp = CONFIG['log_clamp']
    p = jax.nn.sigmoid(inputs).reshape(-1)
    t = targets.reshape(-1)
    intersection = jnp.sum(p * t)
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (jnp.sum(p) + jnp.sum(t) + smooth)
    log_p = jnp.maximum(jnp.log(p), clamp)
    log_1mp = jnp.maximum(jnp.log(1.0 - p), clamp)
    bce = jnp.mean(-(t * log_p + (1.0 - t) * log_1mp))
    return bce + dice_loss
