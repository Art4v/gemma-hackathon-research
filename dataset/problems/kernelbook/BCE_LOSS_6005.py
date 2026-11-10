"""[bf16 variant] BCE loss with logit prior shift: one_hot(labels) -> logits - log(C) -> mean BCE-with-logits -> scale by C."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCE_LOSS_6005',
    'batch': 65536,
    'num_classes': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_logits, k_labels = jax.random.split(key)
    logits = jax.random.uniform(
        k_logits, (CONFIG['batch'], CONFIG['num_classes']), dtype
    )
    labels = jax.random.randint(
        k_labels, (CONFIG['batch'],), 0, CONFIG['num_classes'], jnp.int32
    )
    return logits, labels


def workload(logits, labels):
    num_samples, num_classes = logits.shape[0], logits.shape[1]
    labels = labels.reshape(num_samples)
    targets = jax.nn.one_hot(labels, num_classes, dtype=logits.dtype)
    scale = jnp.asarray(num_classes, logits.dtype)
    shifted = logits - jnp.log(scale)
    elementwise = (1 - targets) * shifted - jax.nn.log_sigmoid(shifted)
    return jnp.mean(elementwise) * scale
