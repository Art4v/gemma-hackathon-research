"""[bf16 variant] RNN cell: concat -> Linear -> sigmoid -> (dropout=identity) -> Linear -> log_softmax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RNN_10326',
    'batch': 4096,
    'n_in': 2048,
    'n_hidden': 2048,
    'n_out': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    b, n_in, n_hidden, n_out = CONFIG['batch'], CONFIG['n_in'], CONFIG['n_hidden'], CONFIG['n_out']
    k = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(k[0], (b, n_in), dtype)
    hidden = jax.random.uniform(k[1], (b, n_hidden), dtype)
    w_i2h = jax.random.normal(k[2], (n_in + n_hidden, n_hidden), dtype) * 0.02
    b_i2h = jax.random.normal(k[3], (n_hidden,), dtype) * 0.02
    w_h2o = jax.random.normal(k[4], (n_hidden, n_out), dtype) * 0.02
    b_h2o = jax.random.normal(k[5], (n_out,), dtype) * 0.02
    return x, hidden, w_i2h, b_i2h, w_h2o, b_h2o


def workload(x, hidden, w_i2h, b_i2h, w_h2o, b_h2o):
    combined = jnp.concatenate([x, hidden], axis=1)
    h = jax.nn.sigmoid(combined @ w_i2h + b_i2h)
    out = jax.nn.log_softmax(h @ w_h2o + b_h2o, axis=1)
    return out, h
