"""[bf16 variant] Linear -> ELU -> Linear -> Sigmoid -> Linear (Generator MLP over last axis)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Generator_13017',
    'x_shape': (8, 128, 128, 512),
    'input_size': 512,
    'hidden_size': 1024,
    'out_size': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, k1, k2, k3, kb1, kb2, kb3 = jax.random.split(jax.random.key(0), 7)
    d_in, d_h, d_out = CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['out_size']
    x = jax.random.uniform(kx, CONFIG['x_shape'], dtype)
    w1 = jax.random.normal(k1, (d_in, d_h), dtype) * 0.02
    b1 = jax.random.normal(kb1, (d_h,), dtype) * 0.02
    w2 = jax.random.normal(k2, (d_h, d_h), dtype) * 0.02
    b2 = jax.random.normal(kb2, (d_h,), dtype) * 0.02
    w3 = jax.random.normal(k3, (d_h, d_out), dtype) * 0.02
    b3 = jax.random.normal(kb3, (d_out,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.elu(x @ w1 + b1)
    h = jax.nn.sigmoid(h @ w2 + b2)
    return h @ w3 + b3
