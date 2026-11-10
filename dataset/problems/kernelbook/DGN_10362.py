"""DGN: linear+relu encoder, two masked dot-product attention blocks (linear+relu v/k/q, bmm, softmax, bmm, linear+relu), linear Q head."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DGN_10362',
    'batch': 8,
    'n_agent': 1024,
    'num_inputs': 1024,
    'hidden_dim': 1024,
    'num_actions': 1024,
    'num_att_linears': 8,
    'mask_penalty': 9000000000000000.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    n = CONFIG['n_agent']
    din = CONFIG['num_inputs']
    h = CONFIG['hidden_dim']
    a = CONFIG['num_actions']
    scale = CONFIG['weight_scale']
    linears = [(din, h)] + [(h, h)] * CONFIG['num_att_linears'] + [(h, a)]
    keys = jax.random.split(jax.random.key(0), 2 + 2 * len(linears))
    x = jax.random.uniform(keys[0], (b, n, din), dtype)
    mask = jax.random.uniform(keys[1], (b, n, n), dtype)
    params = []
    for i, (f_in, f_out) in enumerate(linears):
        params.append(jax.random.normal(keys[2 + 2 * i], (f_in, f_out), dtype) * scale)
        params.append(jax.random.normal(keys[3 + 2 * i], (f_out,), dtype) * scale)
    return (x, mask, *params)


def workload(x, mask,
             enc_w, enc_b,
             a1_v_w, a1_v_b, a1_k_w, a1_k_b, a1_q_w, a1_q_b, a1_o_w, a1_o_b,
             a2_v_w, a2_v_b, a2_k_w, a2_k_b, a2_q_w, a2_q_b, a2_o_w, a2_o_b,
             q_w, q_b):
    penalty = CONFIG['mask_penalty']

    def attention(h, v_w, v_b, k_w, k_b, q_w_, q_b_, o_w, o_b):
        v = jax.nn.relu(h @ v_w + v_b)
        q = jax.nn.relu(h @ q_w_ + q_b_)
        k = jnp.transpose(jax.nn.relu(h @ k_w + k_b), (0, 2, 1))
        att = jax.nn.softmax((q @ k) * mask - penalty * (1 - mask), axis=2)
        out = att @ v
        return jax.nn.relu(out @ o_w + o_b)

    h1 = jax.nn.relu(x @ enc_w + enc_b)
    h2 = attention(h1, a1_v_w, a1_v_b, a1_k_w, a1_k_b, a1_q_w, a1_q_b, a1_o_w, a1_o_b)
    h3 = attention(h2, a2_v_w, a2_v_b, a2_k_w, a2_k_b, a2_q_w, a2_q_b, a2_o_w, a2_o_b)
    return h3 @ q_w + q_b
