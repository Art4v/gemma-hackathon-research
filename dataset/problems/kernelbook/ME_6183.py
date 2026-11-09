"""Mixture-of-experts linear: softmax gating (x @ gw + gb) mixing per-leaf linear projections and biases."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ME_6183',
    'batch': 4096,
    'in_features': 1024,
    'out_features': 1024,
    'n_leaf': 16,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    fin = CONFIG['in_features']
    fout = CONFIG['out_features']
    nl = CONFIG['n_leaf']
    kx, kgw, kgb, kpw, kpb = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (b, fin), dtype)
    gw = jax.random.normal(kgw, (fin, nl), dtype) * 0.02
    gb = jax.random.normal(kgb, (nl,), dtype) * 0.02
    pw = jax.random.normal(kpw, (fout, fin, nl), dtype) * 0.02
    pb = jax.random.normal(kpb, (fout, nl), dtype) * 0.02
    return x, gw, gb, pw, pb


def workload(x, gw, gb, pw, pb):
    # x: (B, in), gw: (in, L), gb: (L,), pw: (out, in, L), pb: (out, L)
    g = jax.nn.softmax(x @ gw + gb, axis=1)  # (B, L)
    # Same sum as PyTorch, but contracts x first so no (B, out, in) intermediate is built.
    t = jnp.einsum('bi,oil->bol', x, pw)  # (B, out, L)
    return jnp.einsum('bol,bl->bo', t, g) + g @ pb.T
