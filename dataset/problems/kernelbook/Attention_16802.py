"""[bf16 variant] Attention_16802: relu(Linear(v)) * relu(Linear(q)) broadcast over objects -> Linear -> softmax over num_obj."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_16802',
    'batch': 64,
    'num_obj': 128,
    'v_features': 2048,
    'q_features': 1024,
    'mid_features': 1024,
    'glimpses': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 5)
    v = jax.random.uniform(ks[0], (c['batch'], c['num_obj'], c['v_features']), dtype)
    q = jax.random.uniform(ks[1], (c['batch'], c['q_features']), dtype)
    w_v = jax.random.normal(ks[2], (c['v_features'], c['mid_features']), dtype) * 0.02
    b_v = jnp.zeros((c['mid_features'],), dtype)
    w_q = jax.random.normal(ks[3], (c['q_features'], c['mid_features']), dtype) * 0.02
    b_q = jnp.zeros((c['mid_features'],), dtype)
    w_l = jax.random.normal(ks[4], (c['mid_features'], c['glimpses']), dtype) * 0.02
    b_l = jnp.zeros((c['glimpses'],), dtype)
    return v, q, w_v, b_v, w_q, b_q, w_l, b_l


def workload(v, q, w_v, b_v, w_q, b_q, w_l, b_l):
    # Weights are the effective weight_norm weights g * V / ||V||, stored (in, out).
    hv = jax.nn.relu(v @ w_v + b_v)
    hq = jax.nn.relu(q @ w_q + b_q)
    x = hv * hq[:, None, :]
    x = x @ w_l + b_l
    return jax.nn.softmax(x, axis=1)
