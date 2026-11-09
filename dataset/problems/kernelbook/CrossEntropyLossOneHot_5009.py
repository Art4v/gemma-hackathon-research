"""CrossEntropyLossOneHot: log_softmax(dim=-1) * labels -> -sum(dim=-1) -> sum(dim=0) / batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CrossEntropyLossOneHot_5009',
    'dim0': 16,
    'dim1': 32,
    'dim2': 128,
    'label_size': 512,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['dim0'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['label_size'])
    k_preds, k_labels = jax.random.split(jax.random.key(0))
    preds = jax.random.uniform(k_preds, shape, dtype)
    labels = jax.random.uniform(k_labels, shape, dtype)
    return preds, labels


def workload(preds, labels):
    batch_size = preds.shape[0]
    log_probs = jax.nn.log_softmax(preds, axis=-1)
    per_row = jnp.sum(-(log_probs * labels), axis=-1)
    return jnp.sum(per_row, axis=0) / batch_size
