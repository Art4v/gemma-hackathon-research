"""[bf16 variant] ModelNet_2470: concat(ob, ac) -> Linear -> ReLU -> Linear -> ReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ModelNet_2470',
    'batch': 64,
    'seq': 4096,
    'obs_dim': 128,
    'act_dim': 128,
    'h1': 512,
    'h2': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 8)
    lead = (c['batch'], c['seq'])
    d_in = c['obs_dim'] + c['act_dim']
    ob = jax.random.uniform(k[0], lead + (c['obs_dim'],), dtype)
    ac = jax.random.uniform(k[1], lead + (c['act_dim'],), dtype)
    w1 = jax.random.normal(k[2], (d_in, c['h1']), dtype) * 0.02
    b1 = jax.random.normal(k[3], (c['h1'],), dtype) * 0.02
    w2 = jax.random.normal(k[4], (c['h1'], c['h2']), dtype) * 0.02
    b2 = jax.random.normal(k[5], (c['h2'],), dtype) * 0.02
    w3 = jax.random.normal(k[6], (c['h2'], c['obs_dim']), dtype) * 0.02
    b3 = jax.random.normal(k[7], (c['obs_dim'],), dtype) * 0.02
    return ob, ac, w1, b1, w2, b2, w3, b3


def workload(ob, ac, w1, b1, w2, b2, w3, b3):
    h = jnp.concatenate([ob, ac], axis=-1)
    h = jax.nn.relu(h @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return h @ w3 + b3
