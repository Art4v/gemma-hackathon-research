"""[bf16 variant] Sigma L1 smooth loss: abs difference -> where(quadratic, linear) -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SigmaL1SmoothLoss_753',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'threshold': 1.0 / 9.0,
    'quadratic_scale': 4.5,
    'linear_offset': 1.0 / 18.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_output, k_target = jax.random.split(key)
    output = jax.random.uniform(k_output, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return output, target


def workload(output, target):
    reg_diff = jnp.abs(target - output)
    reg_loss = jnp.where(
        reg_diff <= CONFIG['threshold'],
        CONFIG['quadratic_scale'] * jnp.power(reg_diff, 2),
        reg_diff - CONFIG['linear_offset'],
    )
    return jnp.mean(reg_loss)
