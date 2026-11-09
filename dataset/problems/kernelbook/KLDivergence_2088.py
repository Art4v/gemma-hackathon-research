"""[bf16 variant] KL divergence distillation loss: softmax(T/tau), log_softmax(S/tau), kl_div batchmean, scaled by tau^2 * loss_weight."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KLDivergence_2088',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'tau': 1.0,
    'loss_weight': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_s, k_t = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    preds_s = jax.random.uniform(k_s, shape, dtype)
    preds_t = jax.random.uniform(k_t, shape, dtype)
    return preds_s, preds_t


def workload(preds_s, preds_t):
    tau = CONFIG['tau']
    log_p_t = jax.nn.log_softmax(preds_t / tau, axis=1)
    p_t = jnp.exp(log_p_t)
    log_p_s = jax.nn.log_softmax(preds_s / tau, axis=1)
    # Where p_t underflows to 0 the term is 0 * finite = 0, the same as torch's xlogy convention.
    pointwise = p_t * (log_p_t - log_p_s)
    loss = jnp.sum(pointwise) / preds_s.shape[0]
    return CONFIG['loss_weight'] * (tau ** 2) * loss
