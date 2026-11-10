"""[bf16 variant] GRUStep: concat -> linear(z, r) -> sigmoid, concat(r*h, x) -> linear -> tanh, gated hidden update."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GRUStep_8339',
    'batch': 16,
    'seq': 1024,
    'hidden_size': 1024,
    'input_size': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    b, s = CONFIG['batch'], CONFIG['seq']
    hs, ins = CONFIG['hidden_size'], CONFIG['input_size']
    k_h, k_x, k_z, k_r, k_t = jax.random.split(jax.random.key(0), 5)
    h_state = jax.random.uniform(k_h, (b, s, hs), dtype)
    x = jax.random.uniform(k_x, (b, s, ins), dtype)
    w_z = jax.random.normal(k_z, (hs + ins, hs), dtype) * 0.02
    w_r = jax.random.normal(k_r, (hs + ins, hs), dtype) * 0.02
    w_t = jax.random.normal(k_t, (hs + ins, hs), dtype) * 0.02
    return h_state, x, w_z, w_r, w_t


def workload(h_state, x, w_z, w_r, w_t):
    hx = jnp.concatenate([h_state, x], axis=-1)
    z = jax.nn.sigmoid(hx @ w_z)
    r = jax.nn.sigmoid(hx @ w_r)
    t = jnp.tanh(jnp.concatenate([r * h_state, x], axis=-1) @ w_t)
    return (1 - z) * h_state + z * t
