"""[bf16 variant] Sigmoid focal classification loss: stable sigmoid BCE * (1 - p_t)^gamma * alpha weighting * weights."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SigmoidFocalClassificationLoss_3256',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'dim3': 128,
    'gamma': 2.0,
    'alpha': 0.25,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    prediction = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    weights = jax.random.uniform(k3, shape, dtype)
    return prediction, target, weights


def workload(prediction, target, weights):
    gamma = CONFIG['gamma']
    alpha = CONFIG['alpha']
    per_entry_cross_ent = (jnp.maximum(prediction, 0) - prediction * target
                           + jnp.log1p(jnp.exp(-jnp.abs(prediction))))
    probs = jax.nn.sigmoid(prediction)
    p_t = target * probs + (1 - target) * (1 - probs)
    modulating_factor = jnp.power(1.0 - p_t, gamma) if gamma else 1.0
    alpha_weight_factor = (target * alpha + (1 - target) * (1 - alpha)) if alpha is not None else 1.0
    return modulating_factor * alpha_weight_factor * per_entry_cross_ent * weights
