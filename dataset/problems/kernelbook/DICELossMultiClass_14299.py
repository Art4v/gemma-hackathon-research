"""Multi-class DICE loss: per-class sum(p*m), sum(p^2), sum(m^2) over HW, dice ratio, summed over the first C-1 classes, 1 - mean over batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DICELossMultiClass_14299',
    'batch': 16,
    'channels': 4,
    'height': 512,
    'width': 512,
    'eps': 1e-07,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    output = jax.random.uniform(k1, shape, dtype)
    input_mask = jax.random.uniform(k2, shape, dtype)
    return output, input_mask


def workload(output, input_mask):
    eps = CONFIG['eps']
    # Like the PyTorch module, use only the first C-1 channels (num_classes = C - 1).
    probs = output[:, :-1]
    mask = input_mask[:, :-1]
    num = jnp.sum(probs * mask, axis=(2, 3))
    den1 = jnp.sum(probs * probs, axis=(2, 3))
    den2 = jnp.sum(mask * mask, axis=(2, 3))
    dice = 2 * ((num + eps) / (den1 + den2 + eps))
    dice_eso = jnp.sum(dice, axis=1)
    return 1 - jnp.sum(dice_eso) / dice_eso.shape[0]
