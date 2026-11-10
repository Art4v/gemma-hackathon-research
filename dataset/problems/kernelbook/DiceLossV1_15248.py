"""[bf16 variant] DiceLossV1: sigmoid -> gather(labels, dim=1) -> 1 - (p(1-p)+eps)/(p(1-p)+1+eps) -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceLossV1_15248',
    'batch': 4096,
    'num_classes': 16384,
    'epsilon': 1e-9,
    'reduction': 'mean',
}


def create_inputs(dtype=jnp.bfloat16):
    k_logits, k_labels = jax.random.split(jax.random.key(0))
    logits = jax.random.normal(k_logits, (CONFIG['batch'], CONFIG['num_classes']), dtype)
    labels = jax.random.randint(k_labels, (CONFIG['batch'],), 0, CONFIG['num_classes'], dtype=jnp.int32)
    return logits, labels


def workload(logits, labels):
    eps = CONFIG['epsilon']
    prob = jax.nn.sigmoid(logits)
    prob = jnp.take_along_axis(prob, labels.reshape(logits.shape[0], -1), axis=1)
    pq = (1 - prob) * prob
    dsc_i = 1 - (pq + eps) / (pq + 1 + eps)
    if CONFIG['reduction'] == 'mean':
        return dsc_i.mean()
    return dsc_i.sum()
