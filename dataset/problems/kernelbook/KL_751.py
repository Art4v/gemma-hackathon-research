"""[bf16 variant] KL distillation loss: log_softmax(y_s / T, dim=1) -> kl_div(sum) * T^2 / batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KL_751',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
    'T': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    y_s = jax.random.uniform(k1, shape, dtype)
    p_t = jax.random.uniform(k2, shape, dtype)
    return y_s, p_t


def workload(y_s, p_t):
    T = CONFIG['T']
    p_s = jax.nn.log_softmax(y_s / T, axis=1)
    # Matches torch kl_div: xlogy(target, target) - target * input, with 0*log(0) = 0.
    positive = p_t > 0
    xlogx = jnp.where(positive, p_t * jnp.log(jnp.where(positive, p_t, 1.0)), 0.0)
    loss = jnp.sum(xlogx - p_t * p_s)
    return loss * (T ** 2) / p_s.shape[0]
