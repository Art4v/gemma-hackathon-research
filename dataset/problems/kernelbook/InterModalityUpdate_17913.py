"""InterModalityUpdate: Linear+ReLU QKV projections -> bidirectional multi-head cross attention (v<->q) -> concat residual -> Linear+ReLU outputs."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'InterModalityUpdate_17913',
    'batch': 16,
    'num_v': 2048,
    'num_q': 2048,
    'v_size': 1024,
    'q_size': 1024,
    'output_size': 1024,
    'num_head': 4,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 6)
    b, nv, nq = c['batch'], c['num_v'], c['num_q']
    vs, qs, o = c['v_size'], c['q_size'], c['output_size']
    v = jax.random.uniform(ks[0], (b, nv, vs), dtype)
    q = jax.random.uniform(ks[1], (b, nq, qs), dtype)
    w_vlin = jax.random.normal(ks[2], (vs, 3 * o), dtype) * 0.02
    b_vlin = jnp.zeros((3 * o,), dtype)
    w_qlin = jax.random.normal(ks[3], (qs, 3 * o), dtype) * 0.02
    b_qlin = jnp.zeros((3 * o,), dtype)
    w_vout = jax.random.normal(ks[4], (o + vs, o), dtype) * 0.02
    b_vout = jnp.zeros((o,), dtype)
    w_qout = jax.random.normal(ks[5], (o + qs, o), dtype) * 0.02
    b_qout = jnp.zeros((o,), dtype)
    return (v, q, w_vlin, b_vlin, w_qlin, b_qlin, w_vout, b_vout, w_qout, b_qout)


def _heads(x, num_head):
    b, n, f = x.shape
    return x.reshape(b, n, num_head, f // num_head)


def _cross_attn(query, key, val, scale):
    # query (B,Nq,H,d), key/val (B,Nk,H,d); softmax over the key axis.
    s = jnp.einsum('bihd,bjhd->bhij', query, key) / scale
    p = jax.nn.softmax(s, axis=-1)
    out = jnp.einsum('bhij,bjhd->bihd', p, val)
    b, n, h, d = out.shape
    return out.reshape(b, n, h * d)


def workload(v, q, w_vlin, b_vlin, w_qlin, b_qlin, w_vout, b_vout, w_qout, b_qout):
    num_head = CONFIG['num_head']
    output_size = w_vlin.shape[1] // 3
    scale = (output_size // num_head) ** 0.5

    v_tran = jax.nn.relu(v @ w_vlin + b_vlin)
    q_tran = jax.nn.relu(q @ w_qlin + b_qlin)
    v_key, v_query, v_val = [_heads(t, num_head) for t in jnp.split(v_tran, 3, axis=2)]
    q_key, q_query, q_val = [_heads(t, num_head) for t in jnp.split(q_tran, 3, axis=2)]

    v_update = _cross_attn(v_query, q_key, q_val, scale)
    q_update = _cross_attn(q_query, v_key, v_val, scale)

    cat_v = jnp.concatenate((v, v_update), axis=2)
    cat_q = jnp.concatenate((q, q_update), axis=2)
    updated_v = jax.nn.relu(cat_v @ w_vout + b_vout)
    updated_q = jax.nn.relu(cat_q @ w_qout + b_qout)
    return updated_v, updated_q
