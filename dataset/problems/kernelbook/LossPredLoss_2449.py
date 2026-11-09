"""LossPredLoss: pairwise flip-difference, sign of target, hinge (clamp 1 - one*pred), sum / (N//2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LossPredLoss_2449',
    'batch': 1024,
    'dim1': 128,
    'dim2': 256,
    'margin': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'])
    pred_loss = jax.random.uniform(k1, shape, dtype)
    target_loss = jax.random.uniform(k2, shape, dtype)
    return pred_loss, target_loss


def workload(pred_loss, target_loss):
    half = pred_loss.shape[0] // 2
    p = (pred_loss - jnp.flip(pred_loss, 0))[:half]
    t = (target_loss - jnp.flip(target_loss, 0))[:half]
    t = jax.lax.stop_gradient(t)
    one = 2 * jnp.sign(jnp.maximum(t, 0)) - 1
    loss = jnp.sum(jnp.maximum(CONFIG['margin'] - one * p, 0))
    return loss / p.shape[0]
