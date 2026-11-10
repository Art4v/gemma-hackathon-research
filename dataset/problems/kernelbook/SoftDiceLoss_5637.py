"""[bf16 variant] SoftDiceLoss: sigmoid -> per-sample flatten -> dice score -> mean(1 - score)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftDiceLoss_5637',
    'batch': 16,
    'channels': 4,
    'height': 512,
    'width': 512,
    'eps': 1e-9,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    logits = jax.random.uniform(k1, shape, dtype)
    targets = jax.random.uniform(k2, shape, dtype)
    return logits, targets


def workload(logits, targets):
    eps = CONFIG['eps']
    num = targets.shape[0]
    m1 = jax.nn.sigmoid(logits).reshape(num, -1)
    m2 = targets.reshape(num, -1).astype(jnp.float32)
    intersection = jnp.sum(m1 * m2, axis=1)
    union = jnp.sum(m1, axis=1) + jnp.sum(m2, axis=1)
    score = (2 * intersection + eps) / (union + eps)
    return jnp.mean(1 - score)
