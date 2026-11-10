"""[bf16 variant] BerHu loss: signed max threshold, abs diff, masked square, mean over concatenated terms."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BerHuLoss_7444',
    'n': 16777216,
    'threshold': 0.2,
    'block': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_pred, k_gt = jax.random.split(key)
    pred = jax.random.uniform(k_pred, (CONFIG['n'],), dtype)
    gt = jax.random.uniform(k_gt, (CONFIG['n'],), dtype)
    return pred, gt


def workload(pred, gt):
    delta = pred - gt
    huber_c = CONFIG['threshold'] * jnp.max(delta)
    diff = jnp.abs(delta)
    mask = jax.lax.stop_gradient(diff > huber_c)
    values = diff + jnp.where(mask, diff ** 2, jnp.zeros_like(diff))
    n = diff.shape[0]
    block = CONFIG['block']
    padded = jnp.pad(values, (0, (-n) % block))
    total = jnp.sum(jnp.sum(padded.reshape(-1, block), axis=1), axis=0)
    return total / (n + jnp.sum(mask))
