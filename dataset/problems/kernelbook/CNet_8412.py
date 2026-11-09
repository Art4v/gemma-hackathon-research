"""[bf16 variant] CNet: Linear(s) + Linear(a) -> ReLU -> Linear(30 -> 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CNet_8412',
    'batch_shape': (8, 128, 128),
    's_dim': 128,
    'a_dim': 128,
    'hidden': 30,
    'out_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 8)
    bshape = CONFIG['batch_shape']
    s_dim, a_dim = CONFIG['s_dim'], CONFIG['a_dim']
    hidden, out_dim = CONFIG['hidden'], CONFIG['out_dim']
    s = jax.random.uniform(keys[0], bshape + (s_dim,), dtype)
    a = jax.random.uniform(keys[1], bshape + (a_dim,), dtype)
    ws = jax.random.normal(keys[2], (s_dim, hidden), dtype) * 0.02
    bs = jax.random.normal(keys[3], (hidden,), dtype) * 0.02
    wa = jax.random.normal(keys[4], (a_dim, hidden), dtype) * 0.02
    ba = jax.random.normal(keys[5], (hidden,), dtype) * 0.02
    wo = jax.random.normal(keys[6], (hidden, out_dim), dtype) * 0.02
    bo = jax.random.normal(keys[7], (out_dim,), dtype) * 0.02
    return s, a, ws, bs, wa, ba, wo, bo


def workload(s, a, ws, bs, wa, ba, wo, bo):
    x = s @ ws + bs
    y = a @ wa + ba
    net = jax.nn.relu(x + y)
    return net @ wo + bo
