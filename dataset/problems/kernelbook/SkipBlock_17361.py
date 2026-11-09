"""[bf16 variant] SkipBlock: act(Linear_in(x)) -> Linear_out + Linear_proj(x) -> act."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SkipBlock_17361',
    'batch': 8,
    'dim1': 32,
    'dim2': 128,
    'input_dim': 1024,
    'output_dim': 1024,
    'activation': 'relu',
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(keys[0], (c['batch'], c['dim1'], c['dim2'], c['input_dim']), dtype)
    w_in = jax.random.normal(keys[1], (c['input_dim'], c['output_dim']), dtype) * 0.02
    b_in = jax.random.normal(keys[2], (c['output_dim'],), dtype) * 0.02
    w_out = jax.random.normal(keys[3], (c['output_dim'], c['output_dim']), dtype) * 0.02
    b_out = jax.random.normal(keys[4], (c['output_dim'],), dtype) * 0.02
    w_proj = jax.random.normal(keys[5], (c['input_dim'], c['output_dim']), dtype) * 0.02
    b_proj = jax.random.normal(keys[6], (c['output_dim'],), dtype) * 0.02
    return x, w_in, b_in, w_out, b_out, w_proj, b_proj


def workload(x, w_in, b_in, w_out, b_out, w_proj, b_proj):
    act = jax.nn.relu if CONFIG['activation'] == 'relu' else jnp.tanh
    y = act(x @ w_in + b_in)
    return act(y @ w_out + b_out + x @ w_proj + b_proj)
