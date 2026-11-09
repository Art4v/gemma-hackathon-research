"""SelfAttn_3792: Linear(D->1) scorer -> softmax over sequence -> score-weighted sum of inputs."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttn_3792',
    'batch': 16,
    'seq_len': 2048,
    'dhid': 1024,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    b, t, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dhid']
    inp = jax.random.uniform(k_x, (b, t, d), dtype)
    w = jax.random.normal(k_w, (d, 1), dtype) * 0.02
    bias = jax.random.normal(k_b, (1,), dtype) * 0.02
    return inp, w, bias


def workload(inp, w, bias):
    scores = jax.nn.softmax(inp @ w + bias, axis=1)  # (B, T, 1)
    cont = jnp.matmul(jnp.swapaxes(scores, 1, 2), inp)  # (B, 1, D)
    return jnp.squeeze(cont, axis=1)
