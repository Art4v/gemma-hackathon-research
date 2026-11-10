"""[bf16 variant] partCE: masked argmax over classes -> cross-entropy (sum reduction) of scores against that index."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'partCE_8260',
    'batch': 16,
    'num_classes': 128,
    'height': 128,
    'width': 128,
    'maximum_score': 100000.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['num_classes'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    scores = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return scores, target


def workload(scores, target):
    par_scores = scores - (1.0 - target) * CONFIG['maximum_score']
    max_ind = jnp.argmax(par_scores, axis=1)  # (N, H, W)
    max_ind = jax.lax.stop_gradient(max_ind)
    logp = jax.nn.log_softmax(scores, axis=1)
    picked = jnp.take_along_axis(logp, max_ind[:, None, :, :], axis=1)[:, 0]
    # size_average=False -> reduction='sum'
    return -jnp.sum(picked)
