"""[bf16 variant] Pinball (quantile) loss: sub -> max(tau*d, (tau-1)*d) -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PinballLoss_6415',
    'batch': 32,
    'channels': 64,
    'height': 128,
    'width': 128,
    'tau': 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    y = jax.random.uniform(k1, shape, dtype)
    y_hat = jax.random.uniform(k2, shape, dtype)
    return y, y_hat


def workload(y, y_hat):
    tau = CONFIG['tau']
    delta = y - y_hat
    pinball = jnp.maximum(tau * delta, (tau - 1.0) * delta)
    return jnp.mean(pinball)
