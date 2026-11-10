"""[bf16 variant] Focal loss: soft-target cross_entropy (log_softmax over class axis, mean reduction) -> exp -> (1 - pt)^gamma * logpt -> balance scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_4521',
    'batch': 8,
    'num_classes': 4,
    'height': 1024,
    'width': 1024,
    'class_axis': 1,
    'focusing_param': 2,
    'balance_param': 0.25,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['num_classes'], CONFIG['height'], CONFIG['width'])
    key_output, key_target = jax.random.split(jax.random.key(0))
    output = jax.random.uniform(key_output, shape, dtype)
    target = jax.random.uniform(key_target, shape, dtype)
    return output, target


def workload(output, target):
    class_axis = CONFIG['class_axis']
    log_probs = jax.nn.log_softmax(output, axis=class_axis)
    cross_entropy = jnp.mean(-jnp.sum(target * log_probs, axis=class_axis))
    logpt = -cross_entropy
    pt = jnp.exp(logpt)
    focal_loss = -((1 - pt) ** CONFIG['focusing_param']) * logpt
    return CONFIG['balance_param'] * focal_loss
