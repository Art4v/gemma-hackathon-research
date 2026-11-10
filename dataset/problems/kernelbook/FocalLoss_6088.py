"""FocalLoss_6088: log_softmax -> soft-target cross-entropy (mean over non-class dims) -> exp(-ce) -> (1 - p)**gamma * ce."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_6088',
    'batch': 16,
    'num_classes': 128,
    'height': 128,
    'width': 128,
    'class_axis': 1,
    'gamma': 2,
    'eps': 1e-07,  # stored by the PyTorch module, unused in forward
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['num_classes'], CONFIG['height'], CONFIG['width'])
    k_input, k_target = jax.random.split(jax.random.key(0))
    logits = jax.random.uniform(k_input, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return logits, target


def workload(logits, target):
    class_axis = CONFIG['class_axis']
    log_probs = jax.nn.log_softmax(logits, axis=class_axis)
    count = logits.size / logits.shape[class_axis]
    ce = -jnp.sum(log_probs * target) / count
    p = jnp.exp(-ce)
    loss = (1.0 - p) ** CONFIG['gamma'] * ce
    return jnp.mean(loss)
