"""Soft-label cross_entropy (mean) -> focal loss -(1-exp(-ce))**gamma * (-ce) * balance."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_16829',
    'batch': 16,
    'num_classes': 128,
    'height': 128,
    'width': 128,
    'focusing_param': 2,
    'balance_param': 0.25,
}


def create_inputs(dtype=jnp.float32):
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['num_classes'], CONFIG['height'], CONFIG['width'])
    output = jax.random.uniform(k_out, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return output, target


def workload(output, target):
    # Float target of same shape => PyTorch probability-target CE, class axis 1,
    # mean over all non-class positions.
    logp = jax.nn.log_softmax(output, axis=1)
    cross_entropy = jnp.mean(-jnp.sum(target * logp, axis=1))
    logpt = -cross_entropy
    pt = jnp.exp(logpt)
    focal_loss = -(1 - pt) ** CONFIG['focusing_param'] * logpt
    return CONFIG['balance_param'] * focal_loss
