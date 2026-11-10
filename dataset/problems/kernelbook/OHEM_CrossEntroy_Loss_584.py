"""OHEM cross-entropy: soft-target CrossEntropy(reduction='none') -> sort desc -> threshold/keep_num select -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'OHEM_CrossEntroy_Loss_584',
    'batch': 16,
    'classes': 128,
    'height': 128,
    'width': 128,
    'threshold': 4.0,
    'keep_num': 4,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['classes'], CONFIG['height'], CONFIG['width'])
    output = jax.random.uniform(k1, shape, dtype)
    # Float target of same shape as output -> PyTorch class-probability mode.
    target = jax.random.uniform(k2, shape, dtype)
    return output, target


def workload(output, target):
    threshold = CONFIG['threshold']
    keep_num = CONFIG['keep_num']
    logp = jax.nn.log_softmax(output, axis=1)
    loss = -jnp.sum(target * logp, axis=1).reshape(-1)
    sorted_desc = -jnp.sort(-loss)
    thr = sorted_desc[keep_num]
    mask = loss > threshold
    mean_above = jnp.sum(jnp.where(mask, loss, 0.0)) / jnp.sum(mask).astype(loss.dtype)
    mean_top = jnp.mean(sorted_desc[:keep_num])
    return jnp.where(thr > threshold, mean_above, mean_top)
