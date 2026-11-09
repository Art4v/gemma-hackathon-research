"""[bf16 variant] Max-pooled query -> nn.MultiheadAttention over the sequence -> repeat output and head-averaged weights T times."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MAXATTN_17392',
    'seq_len': 2048,
    'batch': 8,
    'embed_dim': 1024,
    'num_heads': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    t, b, e = CONFIG['seq_len'], CONFIG['batch'], CONFIG['embed_dim']
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    hidden = jax.random.uniform(k0, (t, b, e), dtype)
    w_in = jax.random.normal(k1, (e, 3 * e), dtype) * 0.02
    b_in = jax.random.normal(k2, (3 * e,), dtype) * 0.02
    w_out = jax.random.normal(k3, (e, e), dtype) * 0.02
    b_out = jax.random.normal(k4, (e,), dtype) * 0.02
    return hidden, w_in, b_in, w_out, b_out


def workload(hidden, w_in, b_in, w_out, b_out):
    # 2D input is PyTorch's unbatched MHA path: treat as batch 1, reshape at the end.
    unbatched = hidden.ndim == 2
    x = hidden[:, None, :] if unbatched else hidden
    t, b, e = x.shape
    h = CONFIG['num_heads']
    d = e // h

    query = jnp.max(x, axis=0)  # (B, E)
    q = (query @ w_in[:, :e] + b_in[:e]).reshape(b, h, d) * (d ** -0.5)
    k = (x @ w_in[:, e:2 * e] + b_in[e:2 * e]).reshape(t, b, h, d)
    v = (x @ w_in[:, 2 * e:] + b_in[2 * e:]).reshape(t, b, h, d)

    p = jax.nn.softmax(jnp.einsum('bhd,tbhd->bht', q, k), axis=-1)
    o = jnp.einsum('bht,tbhd->bhd', p, v).reshape(b, e)
    out = o @ w_out + b_out  # (B, E)
    weight = jnp.mean(p, axis=1)  # (B, T), averaged over heads

    out_rep = jnp.broadcast_to(out[None], (t, b, e))
    w_rep = jnp.broadcast_to(weight[:, None, :], (b, t, t))
    if unbatched:
        return out_rep[:, 0], w_rep.reshape(1, t * t)
    return out_rep, w_rep
