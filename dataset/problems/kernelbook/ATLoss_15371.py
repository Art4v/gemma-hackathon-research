"""[bf16 variant] ATLoss: multilabel categorical cross-entropy (sign flip, 1e12 masking, zero-column logsumexp, mean)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ATLoss_15371',
    'batch': 16,
    'dim1': 64,
    'dim2': 128,
    'num_classes': 256,
    'mask_value': 1000000000000.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['num_classes'])
    k1, k2 = jax.random.split(jax.random.key(0))
    logits = jax.random.uniform(k1, shape, dtype)
    labels = jax.random.uniform(k2, shape, dtype)
    return logits, labels


def workload(logits, labels):
    # Matches torch: forward calls mcc(y_pred=labels, y_true=logits).
    y_pred, y_true = labels, logits
    big = CONFIG['mask_value']
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * big
    y_pred_pos = y_pred - (1 - y_true) * big
    zeros = jnp.zeros_like(y_pred[..., :1])
    y_pred_neg = jnp.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = jnp.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = jax.nn.logsumexp(y_pred_neg, axis=-1)
    pos_loss = jax.nn.logsumexp(y_pred_pos, axis=-1)
    return jnp.mean(neg_loss + pos_loss)
