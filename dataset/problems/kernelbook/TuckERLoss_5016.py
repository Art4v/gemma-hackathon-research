"""[bf16 variant] TuckERLoss: element-wise log of positive scores and of (1 - negative scores), mean, negate, average of the two terms."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TuckERLoss_5016',
    'shape': (8, 128, 128, 128),
    'input_eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_p, k_n = jax.random.split(key)
    eps = CONFIG['input_eps']
    p_score = jax.random.uniform(k_p, CONFIG['shape'], dtype, minval=eps, maxval=1.0 - eps)
    n_score = jax.random.uniform(k_n, CONFIG['shape'], dtype, minval=eps, maxval=1.0 - eps)
    return p_score, n_score


def workload(p_score, n_score):
    p_term = -jnp.mean(jnp.log(p_score))
    n_term = -jnp.mean(jnp.log(1 - n_score))
    return (p_term + n_term) / 2
