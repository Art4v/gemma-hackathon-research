"""[bf16 variant] Concat(dim=1) -> Linear -> LeakyReLU -> Linear (dropout is identity at inference)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '_MLP_C_5185',
    'batch': 8,
    'c1': 64,
    'c2': 64,
    'height': 128,
    'in_size': 512,
    'fil_num': 512,
    'out_size': 2,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 6)
    x1 = jax.random.uniform(k[0], (c['batch'], c['c1'], c['height'], c['in_size']), dtype)
    x2 = jax.random.uniform(k[1], (c['batch'], c['c2'], c['height'], c['in_size']), dtype)
    w1 = jax.random.normal(k[2], (c['in_size'], c['fil_num']), dtype) * 0.02
    b1 = jax.random.normal(k[3], (c['fil_num'],), dtype) * 0.02
    w2 = jax.random.normal(k[4], (c['fil_num'], c['out_size']), dtype) * 0.02
    b2 = jax.random.normal(k[5], (c['out_size'],), dtype) * 0.02
    return x1, x2, w1, b1, w2, b2


def workload(x1, x2, w1, b1, w2, b2):
    x = jnp.concatenate((x1, x2), axis=1)
    h = jax.nn.leaky_relu(x @ w1 + b1, negative_slope=CONFIG['negative_slope'])
    return h @ w2 + b2
