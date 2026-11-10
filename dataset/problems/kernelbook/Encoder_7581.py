"""[bf16 variant] Encoder_7581: reshape -> Linear -> ReLU -> Linear -> ReLU -> (Linear, exp(Linear))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Encoder_7581',
    'outer_batch': 64,
    'inner_batch': 8,
    'alph_size': 128,
    'seq_len': 128,
    'hidden1': 1536,
    'hidden2': 1536,
    'z_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 9)
    in_dim = CONFIG['seq_len'] * CONFIG['alph_size']
    h1 = CONFIG['hidden1']
    h2 = CONFIG['hidden2']
    z = CONFIG['z_dim']
    x = jax.random.uniform(
        keys[0],
        (CONFIG['outer_batch'], CONFIG['inner_batch'], CONFIG['seq_len'], CONFIG['alph_size']),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (in_dim, h1), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (h1,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (h1, h2), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (h2,), dtype) * 0.02
    w_loc = jax.random.normal(keys[5], (h2, z), dtype) * 0.02
    b_loc = jax.random.normal(keys[6], (z,), dtype) * 0.02
    w_scale = jax.random.normal(keys[7], (h2, z), dtype) * 0.02
    b_scale = jax.random.normal(keys[8], (z,), dtype) * 0.02
    return x, w1, b1, w2, b2, w_loc, b_loc, w_scale, b_scale


def workload(x, w1, b1, w2, b2, w_loc, b_loc, w_scale, b_scale):
    flat = jnp.reshape(x, (-1, w1.shape[0]))
    hidden1 = jax.nn.relu(flat @ w1 + b1)
    hidden2 = jax.nn.relu(hidden1 @ w2 + b2)
    z_loc = hidden2 @ w_loc + b_loc
    z_scale = jnp.exp(hidden2 @ w_scale + b_scale)
    return z_loc, z_scale
