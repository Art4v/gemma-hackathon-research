"""[bf16 variant] DGN: Linear+ReLU encoder -> 2x masked dot-product attention block (relu Linear v/q/k, masked softmax, bmm, relu Linear) -> Linear Q head."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DGN_13801',
    'batch': 64,
    'n_agent': 512,
    'num_inputs': 1024,
    'hidden_dim': 1024,
    'num_actions': 128,
    'mask_penalty': 9000000000000000.0,
}


def _linear_params(k, din, dout, dtype):
    kw, kb = jax.random.split(k)
    return (jax.random.normal(kw, (din, dout), dtype) * 0.02,
            jax.random.normal(kb, (dout,), dtype) * 0.02)


def create_inputs(dtype=jnp.bfloat16):
    b, n = CONFIG['batch'], CONFIG['n_agent']
    din, h, a = CONFIG['num_inputs'], CONFIG['hidden_dim'], CONFIG['num_actions']
    keys = jax.random.split(jax.random.key(0), 13)
    x = jax.random.uniform(keys[0], (b, n, din), dtype)
    mask = jax.random.uniform(keys[1], (b, n, n), dtype)
    params = list(_linear_params(keys[2], din, h, dtype))
    for i in range(8):  # att_1 then att_2: v, q, k, out
        params += _linear_params(keys[3 + i], h, h, dtype)
    params += _linear_params(keys[11], h, a, dtype)
    return (x, mask, *params)


def _att(x, mask, vw, vb, qw, qb, kw, kb, ow, ob):
    v = jax.nn.relu(x @ vw + vb)
    q = jax.nn.relu(x @ qw + qb)
    k = jax.nn.relu(x @ kw + kb)
    scores = jnp.einsum('bnh,bmh->bnm', q, k)
    logits = scores * mask - CONFIG['mask_penalty'] * (1 - mask)
    att = jax.nn.softmax(logits, axis=2)
    out = jnp.einsum('bnm,bmh->bnh', att, v)
    return jax.nn.relu(out @ ow + ob)


def workload(x, mask, enc_w, enc_b, *rest):
    att1, att2, (q_w, q_b) = rest[0:8], rest[8:16], rest[16:18]
    h1 = jax.nn.relu(x @ enc_w + enc_b)
    h2 = _att(h1, mask, *att1)
    h3 = _att(h2, mask, *att2)
    return h3 @ q_w + q_b
