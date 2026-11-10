"""Linear -> ReLU -> Linear classifier head (KernelBench _Classifier_4031)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '_Classifier_4031',
    'lead_shape': (64, 64, 64),
    'z_c_dim': 128,
    'hidden_dim': 50,
    'out_dim': 10,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.split(jax.random.key(0), 5)
    z, h, o = CONFIG['z_c_dim'], CONFIG['hidden_dim'], CONFIG['out_dim']
    x = jax.random.uniform(k[0], (*CONFIG['lead_shape'], z), dtype)
    w1 = jax.random.normal(k[1], (z, h), dtype) * 0.02
    b1 = jax.random.normal(k[2], (h,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (h, o), dtype) * 0.02
    b2 = jax.random.normal(k[4], (o,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2
