"""[bf16 variant] RNN cell step: concat(x, h) -> Linear (new hidden) and Linear -> LogSoftmax (output)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RNN_9217',
    'batch': 4096,
    'input_size': 4096,
    'hidden_size': 4096,
    'output_size': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    i = CONFIG['input_size']
    h = CONFIG['hidden_size']
    o = CONFIG['output_size']
    k = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(k[0], (b, i), dtype)
    hidden = jax.random.uniform(k[1], (b, h), dtype)
    w_i2h = jax.random.normal(k[2], (i + h, h), dtype) * 0.02
    b_i2h = jax.random.normal(k[3], (h,), dtype) * 0.02
    w_i2o = jax.random.normal(k[4], (i + h, o), dtype) * 0.02
    b_i2o = jax.random.normal(k[5], (o,), dtype) * 0.02
    return x, hidden, w_i2h, b_i2h, w_i2o, b_i2o


def workload(x, hidden, w_i2h, b_i2h, w_i2o, b_i2o):
    combined = jnp.concatenate([x, hidden], axis=1)
    new_hidden = combined @ w_i2h + b_i2h
    output = jax.nn.log_softmax(combined @ w_i2o + b_i2o, axis=1)
    return output, new_hidden
