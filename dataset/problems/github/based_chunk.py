"""[bf16 variant] Based linear attention (2nd-order Taylor feature map), chunked: cumsum normalizer + intra-chunk masked attn + inter-chunk kv state prefix sums."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'based_chunk',
    'batch': 8,
    'heads': 16,
    'seq_len': 4096,
    'dk': 16,
    'dv': 64,
    'chunk_size': 64,
    'eps': 1e-6,
}

_P = lax.Precision.HIGHEST


def create_inputs(dtype=jnp.bfloat16):
    b, h, t = CONFIG['batch'], CONFIG['heads'], CONFIG['seq_len']
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(kq, (b, h, t, CONFIG['dk']), dtype)
    k = jax.random.normal(kk, (b, h, t, CONFIG['dk']), dtype)
    v = jax.random.normal(kv, (b, h, t, CONFIG['dv']), dtype)
    return q, k, v


def _exclusive_cumsum_chunks(x):
    # cumsum over chunk axis 2, shifted by one so chunk i sees only chunks < i
    x = jnp.cumsum(x, axis=2)
    return jnp.concatenate([jnp.zeros_like(x[:, :, :1]), x[:, :, :-1]], axis=2)


def workload(q, k, v):
    c = CONFIG['chunk_size']
    b, h, l, dk = q.shape
    dv = v.shape[-1]
    n = l // c

    q = q * (dk ** -0.5)
    k_cumsum = jnp.cumsum(k, axis=-2)
    kk_cumsum = jnp.cumsum(k[..., :, None] * k[..., None, :], axis=-3)
    z = jnp.sum(q * k_cumsum, axis=-1)
    z = z + jnp.einsum('bhtx,bhty,bhtxy->bht', q, q, kk_cumsum, precision=_P) * 0.5
    z = z + (jnp.arange(l, dtype=z.dtype) + 1.0)[None, None, :]
    _o = jnp.cumsum(v, axis=-2)

    qc = q.reshape(b, h, n, c, dk)
    kc = k.reshape(b, h, n, c, dk)
    vc = v.reshape(b, h, n, c, dv)

    attn = jnp.einsum('bhncx,bhndx->bhncd', qc, kc, precision=_P)
    attn = attn + 0.5 * attn ** 2
    mask = jnp.tril(jnp.ones((c, c), dtype=bool))
    attn = jnp.where(mask, attn, 0.0)
    o = jnp.einsum('bhncd,bhndz->bhncz', attn, vc, precision=_P)

    kv2 = jnp.einsum('bhncx,bhncy,bhncz->bhnxyz', kc, kc, vc, precision=_P)
    kv2 = _exclusive_cumsum_chunks(kv2)
    o = o + 0.5 * jnp.einsum('bhnxyz,bhncx,bhncy->bhncz', kv2, qc, qc, precision=_P)

    kv1 = jnp.einsum('bhncx,bhncy->bhnxy', kc, vc, precision=_P)
    kv1 = _exclusive_cumsum_chunks(kv1)
    o = o + jnp.einsum('bhnxy,bhncx->bhncy', kv1, qc, precision=_P)

    o = o.reshape(b, h, l, dv) + _o
    return o / (z[..., None] + CONFIG['eps'])
