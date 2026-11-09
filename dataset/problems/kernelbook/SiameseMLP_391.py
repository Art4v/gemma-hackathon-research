"""SiameseMLP_391: shared relu(fc1)->relu(fc2) on two inputs, concat, fc3, sigmoid."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SiameseMLP_391',
    'batch': 131072,
    'in_features': 196,
    'hidden': 160,
    'embed': 10,
    'out_features': 2,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 8)
    x1 = jax.random.uniform(keys[0], (c['batch'], c['in_features']), dtype)
    x2 = jax.random.uniform(keys[1], (c['batch'], c['in_features']), dtype)
    w1 = jax.random.normal(keys[2], (c['in_features'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(keys[3], (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(keys[4], (c['hidden'], c['embed']), dtype) * 0.02
    b2 = jax.random.normal(keys[5], (c['embed'],), dtype) * 0.02
    w3 = jax.random.normal(keys[6], (2 * c['embed'], c['out_features']), dtype) * 0.02
    b3 = jax.random.normal(keys[7], (c['out_features'],), dtype) * 0.02
    return x1, x2, w1, b1, w2, b2, w3, b3


def workload(x1, x2, w1, b1, w2, b2, w3, b3):
    def branch(x):
        x = x.reshape(-1, w1.shape[0])
        h = jax.nn.relu(x @ w1 + b1)
        return jax.nn.relu(h @ w2 + b2)

    x = jnp.concatenate([branch(x1), branch(x2)], axis=1)
    return jax.nn.sigmoid(x @ w3 + b3)
