"""[bf16 variant] CrossEntropy: argmax(target, dim=1) -> log_softmax(input, dim=1) -> NLL mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CrossEntropy_3310',
    'batch': 16,
    'num_classes': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_t = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['num_classes'], CONFIG['height'], CONFIG['width'])
    logits = jax.random.uniform(k_x, shape, dtype)
    target = jax.random.uniform(k_t, shape, dtype)
    return logits, target


def workload(logits, target):
    labels = jnp.argmax(target, axis=1)
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=1)
    picked = jnp.take_along_axis(logp, labels[:, None], axis=1)[:, 0]
    return -jnp.mean(picked)
