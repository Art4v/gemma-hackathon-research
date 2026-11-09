"""ISAB: MAB(inducing points I, X) -> H, then MAB(X, H); multi-head attention + residual + relu(fc_o)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ISAB_4635',
    'batch': 16,
    'set_size': 4096,
    'dim_in': 512,
    'dim_out': 512,
    'num_heads': 4,
    'num_inds': 128,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 10)
    x = jax.random.uniform(keys[0], (c['batch'], c['set_size'], c['dim_in']), dtype)
    ind = jax.random.normal(keys[1], (1, c['num_inds'], c['dim_out']), dtype) * 0.02

    def lin(k, din, dout):
        return jax.random.normal(k, (din, dout), dtype) * 0.02, jnp.zeros((dout,), dtype)

    di, do = c['dim_in'], c['dim_out']
    mab0 = [*lin(keys[2], do, do), *lin(keys[3], di, do), *lin(keys[4], di, do), *lin(keys[5], do, do)]
    mab1 = [*lin(keys[6], di, do), *lin(keys[7], do, do), *lin(keys[8], do, do), *lin(keys[9], do, do)]
    return (x, ind, *mab0, *mab1)


def _mab(q_in, k_in, wq, bq, wk, bk, wv, bv, wo, bo):
    q = q_in @ wq + bq
    k = k_in @ wk + bk
    v = k_in @ wv + bv
    b, nq, dv = q.shape
    nk = k.shape[1]
    h = CONFIG['num_heads']
    ds = dv // h
    qh = q.reshape(b, nq, h, ds)
    kh = k.reshape(b, nk, h, ds)
    vh = v.reshape(b, nk, h, ds)
    # scale uses full dim_V, matching the torch module
    s = jnp.einsum('bqhd,bkhd->bhqk', qh, kh) / (dv ** 0.5)
    a = jax.nn.softmax(s, axis=-1)
    o = (qh + jnp.einsum('bhqk,bkhd->bqhd', a, vh)).reshape(b, nq, dv)
    return o + jax.nn.relu(o @ wo + bo)


def workload(x, ind, *params):
    p0, p1 = params[:8], params[8:]
    ind_b = jnp.broadcast_to(ind, (x.shape[0],) + ind.shape[1:])
    hid = _mab(ind_b, x, *p0)
    return _mab(x, hid, *p1)
