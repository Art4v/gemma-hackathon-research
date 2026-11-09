"""FocalLoss: soft-label cross-entropy (mean) -> p = exp(-ce) -> (1 - p)**gamma * ce -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_16988',
    'batch': 16,
    'num_classes': 128,
    'height': 128,
    'width': 128,
    'gamma': 2,
    'eps': 1e-07,  # ponytail: unused by the original forward, kept for fidelity
}


def create_inputs(dtype=jnp.float32):
    k_in, k_tgt = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['num_classes'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k_in, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return x, target


def workload(x, target):
    # Float target with input's shape -> PyTorch class-probability CE over dim 1,
    # 'mean' reduction averages over the N*H*W positions.
    logprobs = jax.nn.log_softmax(x, axis=1)
    logp = jnp.mean(-jnp.sum(target * logprobs, axis=1))
    p = jnp.exp(-logp)
    loss = (1.0 - p) ** CONFIG['gamma'] * logp
    return jnp.mean(loss)
