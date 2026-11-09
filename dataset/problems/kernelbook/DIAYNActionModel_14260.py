"""Linear -> tanh -> Linear -> per-row policy gather -> softmax (DIAYN action model)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DIAYNActionModel_14260',
    'batch': 16384,
    'n_observations': 1024,
    'n_hidden': 1024,
    'n_policies': 16,
    'n_actions': 128,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_idx, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 6)
    frame = jax.random.uniform(k_x, (c['batch'], c['n_observations']), dtype)
    idx_policy = jax.random.randint(k_idx, (c['batch'],), 0, c['n_policies'], dtype=jnp.int32)
    w1 = jax.random.normal(k_w1, (c['n_observations'], c['n_hidden']), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (c['n_hidden'],), dtype) * 0.02
    # linear2 weight stored (in, n_policies, n_actions); same column order as torch's reshape.
    w2 = jax.random.normal(k_w2, (c['n_hidden'], c['n_policies'], c['n_actions']), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (c['n_policies'], c['n_actions']), dtype) * 0.02
    return frame, idx_policy, w1, b1, w2, b2


def workload(frame, idx_policy, w1, b1, w2, b2):
    z = jnp.tanh(frame @ w1 + b1)
    scores = jnp.einsum('bh,hpa->bpa', z, w2) + b2
    picked = jnp.take_along_axis(scores, idx_policy[:, None, None], axis=1)[:, 0, :]
    return jax.nn.softmax(picked, axis=-1)
