"""LayerNorm_2011: per-axis-1 standardization (mean, sample std with ddof=1, subtract, divide), flattening rank-3 inputs first."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_2011',
    'shape': (16, 128, 128, 128),
    'axis': 1,
    'ddof': 1,
    'flatten_rank': 3,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    activation = jax.random.uniform(key, CONFIG['shape'], dtype)
    return (activation,)


def workload(activation):
    axis = CONFIG['axis']
    ori_shape = None
    if activation.ndim == CONFIG['flatten_rank']:
        ori_shape = activation.shape
        activation = activation.reshape(-1, activation.shape[-1])
    means = jnp.mean(activation, axis=axis, keepdims=True)
    stds = jnp.std(activation, axis=axis, keepdims=True, ddof=CONFIG['ddof'])
    activation = (activation - means) / stds
    if ori_shape is not None:
        activation = activation.reshape(ori_shape)
    return activation
