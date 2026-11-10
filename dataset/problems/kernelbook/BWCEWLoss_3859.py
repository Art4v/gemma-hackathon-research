"""BCEWithLogitsLoss (mean reduction, no pos_weight) -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BWCEWLoss_3859',
    'batch': 32,
    'dim1': 512,
    'dim2': 2048,
    'robust_lambda': 0,
    'confidence_penalty': 0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'])
    preds = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return preds, target


def workload(preds, target):
    x = preds
    t = target.astype(x.dtype)
    # Same numerically stable form as PyTorch's BCEWithLogits.
    loss = jnp.maximum(x, 0) - x * t + jnp.log1p(jnp.exp(-jnp.abs(x)))
    loss = jnp.mean(loss)
    lam = CONFIG['robust_lambda']
    if lam > 0:
        loss = (1 - lam) * loss + lam / 2
    # ponytail: confidence_penalty branch omitted (0 by default; utils is undefined upstream).
    return loss
