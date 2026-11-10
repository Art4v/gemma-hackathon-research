"""[bf16 variant] Attention_12916: Linear(U) on transposed features + Linear(W) on transposed hidden -> tanh -> Linear(A) -> softmax(dim=1) -> weighted sum over dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_12916',
    'batch': 16,  # features dims 0 and 2, hidden dims 0 and 1 (torch broadcast forces these equal)
    'seq': 128,   # features dim 1, hidden dim 2; must be > 1 so squeeze(2) is a no-op as in torch
    'image_features_dim': 512,
    'decoder_hidden_state_dim': 512,
    'attention_dim': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    n, m = CONFIG['batch'], CONFIG['seq']
    d, e = CONFIG['image_features_dim'], CONFIG['decoder_hidden_state_dim']
    att = CONFIG['attention_dim']
    ks = jax.random.split(jax.random.key(0), 8)
    features = jax.random.uniform(ks[0], (n, m, n, d), dtype)
    hidden_state = jax.random.uniform(ks[1], (n, n, m, e), dtype)
    u_w = jax.random.normal(ks[2], (d, att), dtype) * 0.02
    u_b = jax.random.normal(ks[3], (att,), dtype) * 0.02
    w_w = jax.random.normal(ks[4], (e, att), dtype) * 0.02
    w_b = jax.random.normal(ks[5], (att,), dtype) * 0.02
    a_w = jax.random.normal(ks[6], (att, 1), dtype) * 0.02
    a_b = jax.random.normal(ks[7], (1,), dtype) * 0.02
    return features, hidden_state, u_w, u_b, w_w, w_b, a_w, a_b


def workload(features, hidden_state, u_w, u_b, w_w, w_b, a_w, a_b):
    ft = jnp.transpose(features, (0, 2, 1, 3))  # (a, c, b, d)
    u = ft @ u_w + u_b
    w = jnp.transpose(hidden_state @ w_w + w_b, (1, 0, 2, 3))
    combined = jnp.tanh(u + w)
    # torch squeeze(2) is a no-op here (dim 2 != 1), so scores keep the trailing 1.
    scores = combined @ a_w + a_b  # (A0, A1, A2, 1)
    alphas = jax.nn.softmax(scores, axis=1)
    # torch: (ft * alphas.unsqueeze(2)).sum(1) broadcasts to 5D (A0, a, c, b, d);
    # out[i,j,k,l] = sum_m alphas[i,m,k] * ft[m,j,k,l].
    weighted = jnp.einsum('imk,mjkl->ijkl', alphas[..., 0], ft)
    return alphas, weighted
