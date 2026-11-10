"""TokenLabelSoftTargetCrossEntropy: sum(-target * log_softmax(x, -1), -1).mean()."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TokenLabelSoftTargetCrossEntropy_2733',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'num_classes': 256,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_t = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['num_classes'])
    x = jax.random.uniform(k_x, shape, dtype)
    target = jax.random.uniform(k_t, shape, dtype)
    return (x, target)


def workload(x, target):
    # ponytail: target.repeat / [..., 1] branches are dead for equal-shape inputs with C != 2.
    loss = jnp.sum(-target * jax.nn.log_softmax(x, axis=-1), axis=-1)
    return jnp.mean(loss)
