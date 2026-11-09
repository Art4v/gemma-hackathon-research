"""[bf16 variant] DiscrimNet: concat(ob, ac) -> Linear -> tanh -> Linear -> tanh -> Linear(1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiscrimNet_10956',
    'batch': 65536,
    'ob_dim': 512,
    'ac_dim': 512,
    'h1': 1024,
    'h2': 1024,
    'out_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 8)
    b = CONFIG['batch']
    d_in = CONFIG['ob_dim'] + CONFIG['ac_dim']
    h1, h2, out = CONFIG['h1'], CONFIG['h2'], CONFIG['out_dim']
    ob = jax.random.uniform(keys[0], (b, CONFIG['ob_dim']), dtype)
    ac = jax.random.uniform(keys[1], (b, CONFIG['ac_dim']), dtype)
    w1 = jax.random.normal(keys[2], (d_in, h1), dtype) * 0.02
    b1 = jax.random.normal(keys[3], (h1,), dtype) * 0.02
    w2 = jax.random.normal(keys[4], (h1, h2), dtype) * 0.02
    b2 = jax.random.normal(keys[5], (h2,), dtype) * 0.02
    w3 = jax.random.normal(keys[6], (h2, out), dtype) * 0.02
    b3 = jax.random.normal(keys[7], (out,), dtype) * 0.02
    return ob, ac, w1, b1, w2, b2, w3, b3


def workload(ob, ac, w1, b1, w2, b2, w3, b3):
    x = jnp.concatenate([ob, ac], axis=1)
    h = jnp.tanh(x @ w1 + b1)
    h = jnp.tanh(h @ w2 + b2)
    return h @ w3 + b3
