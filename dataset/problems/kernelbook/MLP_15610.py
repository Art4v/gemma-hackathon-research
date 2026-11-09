"""[bf16 variant] Two-branch MLP: Linear(left) || Linear(right) -> concat -> ReLU -> Linear -> ReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MLP_15610',
    'batch': 65536,
    'left_channel': 512,
    'right_channel': 512,
    'branch_hidden': 128,
    'hidden': 256,
    'out_channel': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 10)
    two_h = 2 * c['branch_hidden']
    left = jax.random.uniform(ks[0], (c['batch'], c['left_channel']), dtype)
    right = jax.random.uniform(ks[1], (c['batch'], c['right_channel']), dtype)
    w_left = jax.random.normal(ks[2], (c['left_channel'], c['branch_hidden']), dtype) * 0.02
    b_left = jax.random.normal(ks[3], (c['branch_hidden'],), dtype) * 0.02
    w_right = jax.random.normal(ks[4], (c['right_channel'], c['branch_hidden']), dtype) * 0.02
    b_right = jax.random.normal(ks[5], (c['branch_hidden'],), dtype) * 0.02
    w1 = jax.random.normal(ks[6], (two_h, c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(ks[7], (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(ks[8], (c['hidden'], c['out_channel']), dtype) * 0.02
    b2 = jax.random.normal(ks[9], (c['out_channel'],), dtype) * 0.02
    return left, right, w_left, b_left, w_right, b_right, w1, b1, w2, b2


def workload(left, right, w_left, b_left, w_right, b_right, w1, b1, w2, b2):
    left_res = left @ w_left + b_left
    right_res = right @ w_right + b_right
    tmp = jax.nn.relu(jnp.concatenate([left_res, right_res], axis=1))
    tmp = jax.nn.relu(tmp @ w1 + b1)
    return tmp @ w2 + b2
