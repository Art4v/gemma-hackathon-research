"""[bf16 variant] DistillKL: log_softmax(y_s/T) and softmax(y_t/T) over channels, KL-div sum * T^2 / batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DistillKL_18013',
    'batch': 64,
    'channels': 128,
    'height': 64,
    'width': 64,
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
    log_p_s = jax.nn.log_softmax(y_s / T, axis=1)
    log_p_t = jax.nn.log_softmax(y_t / T, axis=1)
    p_t = jnp.exp(log_p_t)
    # p_t * log_p_t matches torch's xlogy(p_t, p_t): log_p_t is finite, so p_t == 0 gives 0.
    loss = jnp.sum(p_t * (log_p_t - log_p_s))
    return loss * (T ** 2) / y_s.shape[0]
