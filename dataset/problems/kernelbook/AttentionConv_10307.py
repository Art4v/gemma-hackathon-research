"""AttentionConv: 1x1 q/k/v convs -> unfold k,v -> + rel pos -> softmax(q*k) -> weighted sum with v."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentionConv_10307',
    'batch': 512,
    'in_channels': 256,
    'out_channels': 256,
    'height': 12,
    'width': 12,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 6)
    b, cin, cout = CONFIG['batch'], CONFIG['in_channels'], CONFIG['out_channels']
    h, w, k = CONFIG['height'], CONFIG['width'], CONFIG['kernel_size']
    x = jax.random.uniform(keys[0], (b, cin, h, w), dtype)
    w_q = jax.random.normal(keys[1], (cin, cout), dtype) * 0.02
    w_k = jax.random.normal(keys[2], (cin, cout), dtype) * 0.02
    w_v = jax.random.normal(keys[3], (cin, cout), dtype) * 0.02
    rel_h = jax.random.normal(keys[4], (cout // 2, 1, 1, k, 1), dtype) * 0.02
    rel_w = jax.random.normal(keys[5], (cout // 2, 1, 1, 1, k), dtype) * 0.02
    return x, w_q, w_k, w_v, rel_h, rel_w


def _unfold2d(t, k, stride, ho, wo):
    # Matches torch t.unfold(2, k, s).unfold(3, k, s): (B, C, Ho, Wo, Kh, Kw).
    span_h = stride * (ho - 1) + 1
    span_w = stride * (wo - 1) + 1
    rows = jnp.stack([t[:, :, i:i + span_h:stride, :] for i in range(k)], axis=-1)
    return jnp.stack([rows[:, :, :, j:j + span_w:stride, :] for j in range(k)], axis=-1)


def workload(x, w_q, w_k, w_v, rel_h, rel_w):
    pad, stride, groups = CONFIG['padding'], CONFIG['stride'], CONFIG['groups']
    b, _, h, w = x.shape
    cout = w_q.shape[1]
    k = rel_h.shape[-2]
    xp = jnp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho = (h + 2 * pad - k) // stride + 1
    wo = (w + 2 * pad - k) // stride + 1

    q = jnp.einsum('bihw,io->bohw', x, w_q)
    kk = _unfold2d(jnp.einsum('bihw,io->bohw', xp, w_k), k, stride, ho, wo)
    v = _unfold2d(jnp.einsum('bihw,io->bohw', xp, w_v), k, stride, ho, wo)

    half = cout // 2
    kk = jnp.concatenate([kk[:, :half] + rel_h, kk[:, half:] + rel_w], axis=1)

    # Row-major reshape reproduces torch's .contiguous().view, including the
    # memory reinterpretation when (Ho, Wo) != (H, W).
    cg = cout // groups
    kk = kk.reshape(b, groups, cg, h, w, -1)
    v = v.reshape(b, groups, cg, h, w, -1)
    q = q.reshape(b, groups, cg, h, w, 1)

    attn = jax.nn.softmax(q * kk, axis=-1)
    return jnp.sum(attn * v, axis=-1).reshape(b, -1, h, w)
