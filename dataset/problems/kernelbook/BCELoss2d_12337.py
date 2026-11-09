"""[bf16 variant] Sigmoid -> binary cross entropy (mean reduction, log clamped at -100)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCELoss2d_12337',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'log_clamp': -100.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    logits = jax.random.uniform(k1, shape, dtype)
    targets = jax.random.uniform(k2, shape, dtype)
    return logits, targets


def workload(logits, targets):
    probs = jax.nn.sigmoid(logits)
    # PyTorch BCELoss clamps each log term at -100 to avoid -inf.
    log_p = jnp.maximum(jnp.log(probs), CONFIG['log_clamp'])
    log_1mp = jnp.maximum(jnp.log(1.0 - probs), CONFIG['log_clamp'])
    loss = -(targets * log_p + (1.0 - targets) * log_1mp)
    return jnp.mean(loss)
