"""[bf16 variant] MultiLabelCircleLoss: sign-flip mask, concat zero, logsumexp over last dim (neg + pos), mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MultiLabelCircleLoss_15281',
    'shape': (16, 128, 128, 128),
    'inf': 1e12,
    'reduction': 'mean',
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    logits = jax.random.uniform(k1, CONFIG['shape'], dtype)
    labels = jax.random.uniform(k2, CONFIG['shape'], dtype)
    return logits, labels


def workload(logits, labels):
    inf = CONFIG['inf']
    logits = (1 - 2 * labels) * logits
    logits_neg = logits - labels * inf
    logits_pos = logits - (1 - labels) * inf
    zeros = jnp.zeros_like(logits[..., :1])
    logits_neg = jnp.concatenate([logits_neg, zeros], axis=-1)
    logits_pos = jnp.concatenate([logits_pos, zeros], axis=-1)
    loss = jax.nn.logsumexp(logits_neg, axis=-1) + jax.nn.logsumexp(logits_pos, axis=-1)
    if CONFIG['reduction'] == 'mean':
        return jnp.mean(loss)
    return jnp.sum(loss)
