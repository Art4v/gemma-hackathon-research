"""ApplySingleAttention: weight-norm Linear+ReLU on v and q, bilinear v^T atten q per mid channel, weight-norm Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ApplySingleAttention_17095',
    'batch': 32,
    'num_obj': 512,
    'que_len': 256,
    'v_features': 1024,
    'q_features': 512,
    'mid_features': 512,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 9)
    b = CONFIG['batch']
    nv = CONFIG['num_obj']
    nq = CONFIG['que_len']
    dv = CONFIG['v_features']
    dq = CONFIG['q_features']
    mid = CONFIG['mid_features']

    v = jax.random.uniform(keys[0], (b, nv, dv), dtype)
    q = jax.random.uniform(keys[1], (b, nq, dq), dtype)
    atten = jax.random.uniform(keys[2], (b, nv, nq), dtype)

    v_w = jax.random.normal(keys[3], (dv, mid), dtype) * 0.02
    v_g = jnp.ones((), dtype)
    v_b = jax.random.normal(keys[4], (mid,), dtype) * 0.02

    q_w = jax.random.normal(keys[5], (dq, mid), dtype) * 0.02
    q_g = jnp.ones((), dtype)
    q_b = jax.random.normal(keys[6], (mid,), dtype) * 0.02

    a_w = jax.random.normal(keys[7], (mid, mid), dtype) * 0.02
    a_g = jnp.ones((), dtype)
    a_b = jax.random.normal(keys[8], (mid,), dtype) * 0.02

    return (v, q, atten, v_w, v_g, v_b, q_w, q_g, q_b, a_w, a_g, a_b)


def _weight_norm_linear(x, w, g, b):
    w_hat = w * (g / jnp.sqrt(jnp.sum(w * w)))
    return x @ w_hat + b


def workload(v, q, atten, v_w, v_g, v_b, q_w, q_g, q_b, a_w, a_g, a_b):
    v_ = jax.nn.relu(_weight_norm_linear(v, v_w, v_g, v_b))
    q_ = jax.nn.relu(_weight_norm_linear(q, q_w, q_g, q_b))
    v_ = jnp.swapaxes(v_, 1, 2)[:, :, None, :]
    q_ = jnp.swapaxes(q_, 1, 2)[:, :, :, None]
    v_ = jnp.matmul(v_, atten[:, None, :, :])
    h_ = jnp.matmul(v_, q_)
    h_ = jnp.squeeze(h_, axis=(2, 3))
    return _weight_norm_linear(h_[:, None, :], a_w, a_g, a_b)
