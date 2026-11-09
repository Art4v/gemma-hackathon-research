"""[bf16 variant] ResForward: Linear+LeakyReLU on cat(phi1, action), 4 residual blocks (Linear+LeakyReLU, Linear, +x) each re-concatenating action, final Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResForward_17457',
    'batch': 16,
    'seq_len': 1024,
    'feature_size': 1024,
    'action_size': 128,
    'num_res_blocks': 4,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    b, s = CONFIG['batch'], CONFIG['seq_len']
    f, a = CONFIG['feature_size'], CONFIG['action_size']
    nb = CONFIG['num_res_blocks']
    keys = jax.random.split(jax.random.key(0), 8)
    phi1 = jax.random.uniform(keys[0], (b, s, f), dtype)
    action = jax.random.uniform(keys[1], (b, s, a), dtype)
    w_in = jax.random.normal(keys[2], (f + a, f), dtype) * 0.02
    b_in = jax.random.normal(keys[3], (f,), dtype) * 0.02
    w_res = jax.random.normal(keys[4], (nb, 2, f + a, f), dtype) * 0.02
    b_res = jax.random.normal(keys[5], (nb, 2, f), dtype) * 0.02
    w_last = jax.random.normal(keys[6], (f + a, f), dtype) * 0.02
    b_last = jax.random.normal(keys[7], (f,), dtype) * 0.02
    return phi1, action, w_in, b_in, w_res, b_res, w_last, b_last


def _cat_linear(x, action, w, b):
    return jnp.concatenate([x, action], axis=-1) @ w + b


def _leaky(x):
    return jax.nn.leaky_relu(x, negative_slope=CONFIG['negative_slope'])


def workload(phi1, action, w_in, b_in, w_res, b_res, w_last, b_last):
    x = _leaky(_cat_linear(phi1, action, w_in, b_in))
    for i in range(w_res.shape[0]):
        res = _leaky(_cat_linear(x, action, w_res[i, 0], b_res[i, 0]))
        x = _cat_linear(res, action, w_res[i, 1], b_res[i, 1]) + x
    return _cat_linear(x, action, w_last, b_last)
