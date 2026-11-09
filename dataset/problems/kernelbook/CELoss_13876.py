"""[bf16 variant] argmax(one-hot-like label, dim=1) -> softmax cross-entropy over channels -> mean * loss_weight."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CELoss_13876',
    'batch': 16,
    'num_classes': 128,
    'height': 128,
    'width': 128,
    'loss_weight': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['num_classes'], CONFIG['height'], CONFIG['width'])
    k_score, k_label = jax.random.split(jax.random.key(0))
    cls_score = jax.random.uniform(k_score, shape, dtype)
    label = jax.random.uniform(k_label, shape, dtype)
    return cls_score, label


def workload(cls_score, label):
    target = jnp.argmax(label, axis=1)
    logp = jax.nn.log_softmax(cls_score.astype(jnp.float32), axis=1)
    nll = -jnp.take_along_axis(logp, target[:, None], axis=1)[:, 0]
    return CONFIG['loss_weight'] * jnp.mean(nll)
