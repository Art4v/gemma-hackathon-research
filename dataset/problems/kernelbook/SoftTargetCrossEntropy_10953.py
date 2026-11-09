"""SoftTargetCrossEntropy: sum(-target * log_softmax(x, -1), -1).mean()."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftTargetCrossEntropy_10953',
    'batch': 16,
    'dim1': 32,
    'dim2': 128,
    'num_classes': 512,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['num_classes'])
    kx, kt = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, shape, dtype)
    target = jax.random.uniform(kt, shape, dtype)
    return x, target


def workload(x, target):
    n_rep, n = x.shape[0], target.shape[0]
    if n != n_rep:
        target = jnp.tile(target, (n_rep // n,) + (1,) * (target.ndim - 1))
    loss = jnp.sum(-target * jax.nn.log_softmax(x, axis=-1), axis=-1)
    return jnp.mean(loss)
