"""AttentionGRUCell: sigmoid(Wr x + Ur C) -> tanh(W x + r * U C) -> gated blend with C."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AttentionGRUCell_12684',
    'batch': 8,
    'dim1': 32,
    'seq': 128,
    'input_size': 1024,
    'hidden_size': 1024,
}


def create_inputs(dtype=jnp.float32):
    b, d1, s = CONFIG['batch'], CONFIG['dim1'], CONFIG['seq']
    i, h = CONFIG['input_size'], CONFIG['hidden_size']
    ks = jax.random.split(jax.random.key(0), 11)
    fact = jax.random.uniform(ks[0], (b, d1, s, i), dtype)
    c = jax.random.uniform(ks[1], (b, d1, s, h), dtype)
    g = jax.random.uniform(ks[2], (b, s, h), dtype)
    wr = jax.random.normal(ks[3], (i, h), dtype) * 0.02
    br = jax.random.normal(ks[4], (h,), dtype) * 0.02
    ur = jax.random.normal(ks[5], (h, h), dtype) * 0.02
    bur = jax.random.normal(ks[6], (h,), dtype) * 0.02
    w = jax.random.normal(ks[7], (i, h), dtype) * 0.02
    bw = jax.random.normal(ks[8], (h,), dtype) * 0.02
    u = jax.random.normal(ks[9], (h, h), dtype) * 0.02
    bu = jax.random.normal(ks[10], (h,), dtype) * 0.02
    return fact, c, g, wr, br, ur, bur, w, bw, u, bu


def _linear(x, w, b):
    return jnp.matmul(x, w, precision=lax.Precision.HIGHEST) + b


def workload(fact, c, g, wr, br, ur, bur, w, bw, u, bu):
    r = jax.nn.sigmoid(_linear(fact, wr, br) + _linear(c, ur, bur))
    h_tilda = jnp.tanh(_linear(fact, w, bw) + r * _linear(c, u, bu))
    # torch: g.unsqueeze(1).expand_as(h_tilda)
    g = jnp.broadcast_to(jnp.expand_dims(g, 1), h_tilda.shape)
    return g * h_tilda + (1 - g) * c
