"""[bf16 variant] RNN cell step: concat -> Linear(i2h), Linear(i2o) -> concat -> Linear(o2o)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RNN_10187',
    'batch': 4096,
    'category_size': 1024,
    'input_size': 4096,
    'hidden_size': 4096,
    'output_size': 2048,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    c, i, h, o = (CONFIG['category_size'], CONFIG['input_size'],
                  CONFIG['hidden_size'], CONFIG['output_size'])
    ks = jax.random.split(jax.random.key(0), 9)
    category = jax.random.uniform(ks[0], (b, c), dtype)
    inp = jax.random.uniform(ks[1], (b, i), dtype)
    hidden = jax.random.uniform(ks[2], (b, h), dtype)
    w_i2h = jax.random.normal(ks[3], (c + i + h, h), dtype) * 0.02
    b_i2h = jax.random.normal(ks[4], (h,), dtype) * 0.02
    w_i2o = jax.random.normal(ks[5], (c + i + h, o), dtype) * 0.02
    b_i2o = jax.random.normal(ks[6], (o,), dtype) * 0.02
    w_o2o = jax.random.normal(ks[7], (h + o, o), dtype) * 0.02
    b_o2o = jax.random.normal(ks[8], (o,), dtype) * 0.02
    return category, inp, hidden, w_i2h, b_i2h, w_i2o, b_i2o, w_o2o, b_o2o


def workload(category, inp, hidden, w_i2h, b_i2h, w_i2o, b_i2o, w_o2o, b_o2o):
    combined = jnp.concatenate((category, inp, hidden), axis=1)
    new_hidden = combined @ w_i2h + b_i2h
    output = combined @ w_i2o + b_i2o
    output = jnp.concatenate((new_hidden, output), axis=1) @ w_o2o + b_o2o
    return output, new_hidden
