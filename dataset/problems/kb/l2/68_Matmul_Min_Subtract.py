"""Matrix multiplication (Linear), elementwise minimum with a constant, and constant subtraction."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '68_Matmul_Min_Subtract',
    'batch_size': 512,
    'in_features': 8192,
    'out_features': 8192,
    'constant': 2.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    batch_size = CONFIG['batch_size']
    in_features = CONFIG['in_features']
    out_features = CONFIG['out_features']

    x = jax.random.uniform(k1, (batch_size, in_features), dtype=dtype)
    weight = jax.random.normal(k2, (in_features, out_features), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (out_features,), dtype=dtype) * 0.02
    constant = jnp.array(CONFIG['constant'], dtype=dtype)
    return x, weight, bias, constant


def workload(x, weight, bias, constant):
    x = jnp.matmul(x, weight) + bias
    x = jnp.minimum(x, constant)
    x = x - constant
    return x
