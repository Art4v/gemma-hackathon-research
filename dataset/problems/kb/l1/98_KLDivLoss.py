"""KLDivLoss: kl_div(log(predictions), targets, reduction='batchmean')."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '98_KLDivLoss',
    'batch': 4096,
    'dim': 8192,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_scale, k_pred, k_tgt = jax.random.split(key, 3)
    b, d = CONFIG['batch'], CONFIG['dim']
    scale = jax.random.uniform(k_scale, (), dtype=dtype)
    predictions = jax.nn.softmax(jax.random.uniform(k_pred, (b, d), dtype=dtype) * scale, axis=-1)
    targets = jax.nn.softmax(jax.random.uniform(k_tgt, (b, d), dtype=dtype), axis=-1)
    return predictions, targets


def workload(predictions, targets):
    log_p = jnp.log(predictions)
    t_log_t = jnp.where(targets > 0, targets * jnp.log(jnp.where(targets > 0, targets, 1.0)), 0.0)
    pointwise = t_log_t - targets * log_p
    return jnp.sum(pointwise) / predictions.shape[0]
