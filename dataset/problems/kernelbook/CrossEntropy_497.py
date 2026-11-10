"""Soft-label cross entropy: log_softmax over channels, -sum(target * logp), mean over N*H*W."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CrossEntropy_497',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    k_score, k_target = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    score = jax.random.uniform(k_score, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return score, target


def workload(score, target):
    # ponytail: the module's upsample branch is skipped because C == H == W in CONFIG,
    # so (ph, pw) == (target.size(1), target.size(2)) and PyTorch never takes it.
    logp = jax.nn.log_softmax(score, axis=1)
    per_pos = -jnp.sum(target * logp, axis=1)
    return jnp.mean(per_pos)
