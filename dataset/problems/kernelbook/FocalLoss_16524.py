"""[bf16 variant] Focal loss: sigmoid -> stack [1-p, p] -> gather by target -> -(1-pt)^gamma * log(pt)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_16524',
    'batch_size': 4096,
    'n_pts': 8192,
    'gamma': 0.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_pred, k_target = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch_size'], CONFIG['n_pts'])
    pred = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return pred, target


def workload(pred, target):
    pos = jax.nn.sigmoid(pred)
    neg = 1 - pos
    pt_all = jnp.stack([neg, pos], axis=-1)
    # .long() truncates toward zero; uniform [0,1) targets all map to index 0.
    index = target.astype(jnp.int32)[..., None]
    pt = jnp.take_along_axis(pt_all, index, axis=-1)[..., 0]
    logpt = jnp.log(pt)
    return -1 * (1 - pt) ** CONFIG['gamma'] * logpt
