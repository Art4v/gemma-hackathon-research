"""Binary focal loss: sigmoid -> p_t select -> -(1 - p_t)^gamma * log(p_t) -> alpha weighting -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_1404',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'alpha': 0.25,
    'gamma': 2,
    'with_logits': True,
    'reduction': 'mean',
    'positive_label': 1,
    'target_threshold': 0.5,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_input, k_target = jax.random.split(key)
    x = jax.random.uniform(k_input, shape, dtype)
    target = (jax.random.uniform(k_target, shape, dtype) >= CONFIG['target_threshold']).astype(dtype)
    return x, target


def workload(x, target):
    alpha = CONFIG['alpha']
    gamma = CONFIG['gamma']
    reduction = CONFIG['reduction']
    p = jax.nn.sigmoid(x) if CONFIG['with_logits'] else x
    is_positive = target == CONFIG['positive_label']
    p_t = jnp.where(is_positive, p, 1 - p)
    fl = -1 * (1 - p_t) ** gamma * jnp.log(p_t)
    fl = jnp.where(is_positive, fl * alpha, fl * (1 - alpha))
    if reduction == 'mean':
        return jnp.mean(fl)
    if reduction == 'sum':
        return jnp.sum(fl)
    return fl
