"""Cross-entropy loss: argmax(target, dim=1) -> NLLLoss(log(output)) mean (BCELoss when C == 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CELoss_7357',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'bce_log_clamp': -100.0,
}


def create_inputs(dtype=jnp.float32):
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    output = jax.random.uniform(k_out, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return output, target


def workload(output, target):
    if target.shape[1] == 1:
        # torch BCELoss clamps each log term at -100.
        clamp = CONFIG['bce_log_clamp']
        log_o = jnp.maximum(jnp.log(output), clamp)
        log_1mo = jnp.maximum(jnp.log(1.0 - output), clamp)
        return -jnp.mean(target * log_o + (1.0 - target) * log_1mo)
    idx = jnp.argmax(target, axis=1)[:, None]
    picked = jnp.take_along_axis(jnp.log(output), idx, axis=1)
    return -jnp.mean(picked)
