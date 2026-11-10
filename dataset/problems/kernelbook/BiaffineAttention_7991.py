"""BiaffineAttention: Linear enc/dec -> bilinear (U) + linear (W_e, W_d) + bias -> softmax over teachers."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BiaffineAttention_7991',
    'batch': 1024,
    'num_teachers': 128,  # also num_labels: einsum 'nd,bnd->bn' requires them equal
    'input_size_encoder': 512,
    'input_size_decoder': 512,
    'hidden_size': 256,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    N = CONFIG['num_teachers']
    De = CONFIG['input_size_encoder']
    Dd = CONFIG['input_size_decoder']
    H = CONFIG['hidden_size']
    ks = jax.random.split(jax.random.key(0), 10)
    input_s = jax.random.uniform(ks[0], (B, De), dtype)
    input_t = jax.random.uniform(ks[1], (B, N, Dd), dtype)
    w_enc = jax.random.normal(ks[2], (De, H), dtype) * 0.02
    b_enc = jax.random.normal(ks[3], (H,), dtype) * 0.02
    w_dec = jax.random.normal(ks[4], (Dd, H), dtype) * 0.02
    b_dec = jax.random.normal(ks[5], (H,), dtype) * 0.02
    W_d = jax.random.normal(ks[6], (N, H), dtype) * 0.02
    W_e = jax.random.normal(ks[7], (N, H), dtype) * 0.02
    b = jnp.zeros((1, N), dtype)
    U = jax.random.normal(ks[8], (N, H, H), dtype) * 0.02
    return input_s, input_t, w_enc, b_enc, w_dec, b_dec, W_d, W_e, b, U


def workload(input_s, input_t, w_enc, b_enc, w_dec, b_dec, W_d, W_e, b, U):
    s = input_s @ w_enc + b_enc                       # (B, H)
    t = input_t @ w_dec + b_dec                       # (B, N, H)
    out_e = s @ W_e.T                                 # (B, N)
    out_d = jnp.einsum('nd,bnd->bn', W_d, t)          # (B, N)
    bil = jnp.einsum('bd,nde->bne', s, U)             # (B, N, H)
    bil = jnp.einsum('bne,bne->bn', bil, t)           # (B, N)
    out = bil + out_d + out_e + b
    return jax.nn.softmax(out, axis=1)
