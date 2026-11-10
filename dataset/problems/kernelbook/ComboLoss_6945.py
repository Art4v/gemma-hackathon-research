"""[bf16 variant] ComboLoss: 0.1*BCEWithLogits(max-pooled logits, presence targets) + BCEWithLogits(logits, targets) + 0*MultiLabelSoftDice."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ComboLoss_6945',
    'batch': 16,
    'num_classes': 8,
    'height': 512,
    'width': 512,
    'weights': (0.1, 0.0, 1.0),
    'dice_eps': 1e-9,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['num_classes'], CONFIG['height'], CONFIG['width'])
    logits = jax.random.uniform(k1, shape, dtype)
    targets = jax.random.uniform(k2, shape, dtype)
    return logits, targets


def _bce_with_logits(x, y):
    return jnp.mean(jnp.maximum(x, 0) - x * y + jnp.log1p(jnp.exp(-jnp.abs(x))))


def workload(logits, targets):
    w_fc, w_dice, w_bce = CONFIG['weights']
    eps = CONFIG['dice_eps']
    b, c = targets.shape[0], targets.shape[1]
    lg = logits.reshape(b, c, -1)
    tg = targets.reshape(b, c, -1)

    targets_fc = (jnp.sum(tg, axis=-1) > 0).astype(logits.dtype)
    logits_fc = jnp.max(lg, axis=-1)
    loss_fc = w_fc * _bce_with_logits(logits_fc, targets_fc)

    # activation=None, so p = logits; dice applies sigmoid itself.
    probs = jax.nn.sigmoid(lg)
    inter = jnp.sum(probs * tg, axis=-1)
    union = jnp.sum(probs, axis=-1) + jnp.sum(tg, axis=-1)
    score = (2 * inter + eps) / (union + eps)
    dice = jnp.mean(1 - score)  # mean over batch, then over classes
    loss_dice = w_dice * dice

    loss_bce = w_bce * _bce_with_logits(logits, targets)
    return loss_fc + loss_bce + loss_dice
