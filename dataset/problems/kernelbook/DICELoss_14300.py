"""DICELoss: conditional squeeze(1), product, sequential sums over axes 2 and 1, Dice ratio, 1 - sum/batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DICELoss_14300',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'eps': 1e-08,
    'squeeze_axis': 1,
    'sum_axes': (2, 1),
    'dice_scale': 2.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_output, k_mask = jax.random.split(jax.random.key(0))
    output = jax.random.uniform(k_output, shape, dtype)
    mask = jax.random.uniform(k_mask, shape, dtype)
    return output, mask


def workload(output, mask):
    squeeze_axis = CONFIG['squeeze_axis']
    first_axis, second_axis = CONFIG['sum_axes']
    eps = CONFIG['eps']

    # torch.squeeze(x, dim) is a no-op when that dim is not of size 1.
    probs = jnp.squeeze(output, squeeze_axis) if output.shape[squeeze_axis] == 1 else output
    target = jnp.squeeze(mask, squeeze_axis) if mask.shape[squeeze_axis] == 1 else mask

    intersection = jnp.sum(jnp.sum(probs * target, axis=first_axis), axis=second_axis)
    den1 = jnp.sum(jnp.sum(probs * probs, axis=first_axis), axis=second_axis)
    den2 = jnp.sum(jnp.sum(target * target, axis=first_axis), axis=second_axis)

    dice = CONFIG['dice_scale'] * ((intersection + eps) / (den1 + den2 + eps))
    return 1 - jnp.sum(dice) / dice.shape[0]
