"""[bf16 variant] Knowledge distillation loss: log_softmax(stu/T) and softmax(tea/T) over dim 1, KL div summed, * T^2 / batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KDLoss_13974',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'T': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_stu, k_tea = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    stu_pred = jax.random.uniform(k_stu, shape, dtype)
    tea_pred = jax.random.uniform(k_tea, shape, dtype)
    return stu_pred, tea_pred


def workload(stu_pred, tea_pred):
    t_scale = CONFIG['T']
    log_s = jax.nn.log_softmax(stu_pred / t_scale, axis=1)
    log_t = jax.nn.log_softmax(tea_pred / t_scale, axis=1)
    t = jnp.exp(log_t)
    # kl_div(reduction='sum'): sum target * (log target - input); t == 0 yields 0 like torch's xlogy.
    kl = jnp.sum(t * (log_t - log_s))
    return kl * t_scale ** 2 / stu_pred.shape[0]
