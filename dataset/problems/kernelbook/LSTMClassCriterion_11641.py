"""[bf16 variant] LSTMClassCriterion: masked NLL gather loss + masked argmax accuracy over (B, T, V) log-probs."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LSTMClassCriterion_11641',
    'batch': 64,
    'seq_len': 128,
    'vocab': 8192,
}


def create_inputs(dtype=jnp.bfloat16):
    k_pred, k_tgt, k_mask = jax.random.split(jax.random.key(0), 3)
    b, t, v = CONFIG['batch'], CONFIG['seq_len'], CONFIG['vocab']
    pred = jax.random.normal(k_pred, (b, t, v), dtype)
    target = jax.random.randint(k_tgt, (b, t), 0, v, dtype=jnp.int32)
    mask = jax.random.uniform(k_mask, (b, t), dtype)
    return pred, target, mask


def workload(pred, target, mask):
    t, v = pred.shape[1], pred.shape[2]
    target = target[:, :t].reshape(-1)
    mask = mask[:, :t].reshape(-1)
    p = pred.reshape(-1, v)
    picked = jnp.take_along_axis(p, target[:, None], axis=1)[:, 0]
    mask_sum = jnp.sum(mask)
    loss = jnp.sum(-picked * mask) / mask_sum
    idx = jnp.argmax(p, axis=1)
    correct = (idx == target).astype(p.dtype) * mask
    accuracy = jnp.sum(correct) / mask_sum
    return loss, accuracy
