"""[bf16 variant] DistillKL: log_softmax(y_s/T) vs softmax(y_t/T) over channels, summed KL div * T^2 / batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DistillKL_404',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'T': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_s, k_t = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    y_s = jax.random.uniform(k_s, shape, dtype)
    y_t = jax.random.uniform(k_t, shape, dtype)
    return y_s, y_t


def workload(y_s, y_t):
    T = CONFIG['T']
    p_s = jax.nn.log_softmax(y_s / T, axis=1)
    log_p_t = jax.nn.log_softmax(y_t / T, axis=1)
    p_t = jnp.exp(log_p_t)
    # kl_div(input=p_s, target=p_t, reduction='sum') = sum(p_t * (log p_t - p_s))
    kl = jnp.sum(p_t * (log_p_t - p_s))
    return kl * (T ** 2) / y_s.shape[0]
