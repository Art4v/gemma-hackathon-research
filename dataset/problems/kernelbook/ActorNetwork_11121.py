"""ActorNetwork forward: Linear-ReLU-Linear-ReLU, then Linear mu head and clamped Linear sigma head."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ActorNetwork_11121',
    'batch': 8,
    'height': 128,
    'width': 128,
    'input_dims': 256,
    'fc1_dims': 512,
    'fc2_dims': 512,
    'n_actions': 128,
    'sigma_min': 1e-6,
    'sigma_max': 1.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 9)
    scale = CONFIG['weight_scale']
    input_dims = CONFIG['input_dims']
    fc1_dims = CONFIG['fc1_dims']
    fc2_dims = CONFIG['fc2_dims']
    n_actions = CONFIG['n_actions']
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], input_dims),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (input_dims, fc1_dims), dtype) * scale
    b1 = jax.random.normal(keys[2], (fc1_dims,), dtype) * scale
    w2 = jax.random.normal(keys[3], (fc1_dims, fc2_dims), dtype) * scale
    b2 = jax.random.normal(keys[4], (fc2_dims,), dtype) * scale
    w_mu = jax.random.normal(keys[5], (fc2_dims, n_actions), dtype) * scale
    b_mu = jax.random.normal(keys[6], (n_actions,), dtype) * scale
    w_sigma = jax.random.normal(keys[7], (fc2_dims, n_actions), dtype) * scale
    b_sigma = jax.random.normal(keys[8], (n_actions,), dtype) * scale
    return x, w1, b1, w2, b2, w_mu, b_mu, w_sigma, b_sigma


def workload(x, w1, b1, w2, b2, w_mu, b_mu, w_sigma, b_sigma):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    mu = h @ w_mu + b_mu
    sigma = jnp.clip(h @ w_sigma + b_sigma, CONFIG['sigma_min'], CONFIG['sigma_max'])
    return mu, sigma
