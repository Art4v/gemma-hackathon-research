"""SoftTargetCrossEntropy: log_softmax(-1) -> -(targets * lsm).sum(-1) -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftTargetCrossEntropy_9484',
    'batch': 16,
    'dim1': 128,
    'dim2': 128,
    'num_classes': 128,
}


def create_inputs(dtype=jnp.float32):
    k_in, k_tgt = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['num_classes'])
    inputs = jax.random.uniform(k_in, shape, dtype)
    targets = jax.random.uniform(k_tgt, shape, dtype)
    return inputs, targets


def workload(inputs, targets):
    lsm = jax.nn.log_softmax(inputs, axis=-1)
    loss = -jnp.sum(targets * lsm, axis=-1)
    return jnp.mean(loss)
