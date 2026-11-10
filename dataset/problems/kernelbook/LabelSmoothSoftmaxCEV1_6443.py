"""Label-smoothed softmax cross-entropy: log_softmax -> smoothed one-hot targets -> ignore-masked sum -> mean over valid."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LabelSmoothSoftmaxCEV1_6443',
    'batch': 8,
    'num_classes': 128,
    'height': 256,
    'width': 256,
    'lb_smooth': 0.1,
    'ignore_index': -100,
    'reduction': 'mean',
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_logits, k_label = jax.random.split(key)
    batch = CONFIG['batch']
    num_classes = CONFIG['num_classes']
    height = CONFIG['height']
    width = CONFIG['width']
    logits = jax.random.normal(k_logits, (batch, num_classes, height, width), dtype)
    label = jax.random.randint(k_label, (batch, height, width), -1, num_classes, dtype=jnp.int32)
    label = jnp.where(label < 0, CONFIG['ignore_index'], label)
    return logits, label


def workload(logits, label):
    logits = logits.astype(jnp.float32)
    num_classes = logits.shape[1]
    ignore = label == CONFIG['ignore_index']
    n_valid = jnp.sum(~ignore)
    clean_label = jnp.where(ignore, 0, label)
    lb_pos = jnp.asarray(1.0 - CONFIG['lb_smooth'], logits.dtype)
    lb_neg = jnp.asarray(CONFIG['lb_smooth'] / num_classes, logits.dtype)
    hot = jax.nn.one_hot(clean_label, num_classes, axis=1, dtype=jnp.bool_)
    targets = jnp.where(hot, lb_pos, lb_neg)
    logs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.sum(logs * targets, axis=1)
    loss = jnp.where(ignore, jnp.zeros_like(loss), loss)
    if CONFIG['reduction'] == 'mean':
        loss = jnp.sum(loss) / n_valid
    if CONFIG['reduction'] == 'sum':
        loss = jnp.sum(loss)
    return loss
