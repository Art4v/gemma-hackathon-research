"""[bf16 variant] Weighted BCE-with-logits loss: clamp(x,0) - x*gt + log(1+exp(-|x|)), weighted sum / sum(w)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'WeightedBCELoss2d_3699',
    'batch': 16,
    'channels': 8,
    'height': 512,
    'width': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    logits = jax.random.uniform(k1, shape, dtype)
    labels = jax.random.uniform(k2, shape, dtype)
    weights = jax.random.uniform(k3, shape, dtype)
    return logits, labels, weights


def workload(logits, labels, weights):
    w = weights.reshape(-1)
    x = logits.reshape(-1)
    gt = labels.reshape(-1)
    loss = jnp.maximum(x, 0) - x * gt + jnp.log(1 + jnp.exp(-jnp.abs(x)))
    loss = loss * w
    return loss.sum() / w.sum()
