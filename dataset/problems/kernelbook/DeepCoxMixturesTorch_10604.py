"""DeepCoxMixtures: two bias-free linear projections, one clamped (hazard ratios) and one log-softmax over axis 1 (gate)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DeepCoxMixturesTorch_10604',
    'batch': 8,
    'height': 128,
    'width': 128,
    'inputdim': 256,
    'k': 128,
    'gamma': 1.0,
    'softmax_axis': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_expert, k_gate = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['inputdim']),
        dtype,
    )
    w_shape = (CONFIG['inputdim'], CONFIG['k'])
    w_expert = jax.random.normal(k_expert, w_shape, dtype) * CONFIG['weight_scale']
    w_gate = jax.random.normal(k_gate, w_shape, dtype) * CONFIG['weight_scale']
    return x, w_expert, w_gate


def workload(x, w_expert, w_gate):
    gamma = CONFIG['gamma']
    log_hazard_ratios = jnp.clip(x @ w_expert, -gamma, gamma)
    log_gate_prob = jax.nn.log_softmax(x @ w_gate, axis=CONFIG['softmax_axis'])
    return log_gate_prob, log_hazard_ratios
