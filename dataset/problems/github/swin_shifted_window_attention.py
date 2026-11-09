"""[bf16 variant] Swin shifted-window attention: roll -> window partition -> qkv -> MHSA + rel-pos bias + shift mask -> proj -> window reverse -> roll back."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'swin_shifted_window_attention',
    'batch': 8,
    'height': 224,
    'width': 224,
    'dim': 192,
    'num_heads': 6,
    'window_size': 7,
    'shift_size': 3,
    'mask_value': -100.0,
}

_HI = lax.Precision.HIGHEST


def create_inputs(dtype=jnp.bfloat16):
    b, h, w, c = CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['dim']
    ws, nh = CONFIG['window_size'], CONFIG['num_heads']
    k = jax.random.split(jax.random.key(0), 6)
    x = jax.random.normal(k[0], (b, h, w, c), dtype)
    qkv_w = jax.random.normal(k[1], (c, 3 * c), dtype) * 0.02
    qkv_b = jax.random.normal(k[2], (3 * c,), dtype) * 0.02
    proj_w = jax.random.normal(k[3], (c, c), dtype) * 0.02
    proj_b = jax.random.normal(k[4], (c,), dtype) * 0.02
    table = jax.random.normal(k[5], ((2 * ws - 1) ** 2, nh), dtype) * 0.02
    return x, qkv_w, qkv_b, proj_w, proj_b, table


def _partition(x, ws):
    b, h, w, c = x.shape
    x = x.reshape(b, h // ws, ws, w // ws, ws, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws, ws, c)


def _reverse(win, ws, b, h, w):
    c = win.shape[-1]
    x = win.reshape(b, h // ws, w // ws, ws, ws, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h, w, c)


def _rel_index(ws):
    ar = jnp.arange(ws)
    coords = jnp.stack(jnp.meshgrid(ar, ar, indexing='ij')).reshape(2, -1)
    rel = coords[:, :, None] - coords[:, None, :]
    return ((rel[0] + ws - 1) * (2 * ws - 1) + (rel[1] + ws - 1)).reshape(-1)


def _attn_mask(h, w, ws, shift, dtype):
    hi = jnp.arange(h)
    wi = jnp.arange(w)
    h_reg = (hi >= h - ws).astype(jnp.int32) + (hi >= h - shift).astype(jnp.int32)
    w_reg = (wi >= w - ws).astype(jnp.int32) + (wi >= w - shift).astype(jnp.int32)
    img = (h_reg[:, None] * 3 + w_reg[None, :]).astype(dtype)
    mw = img.reshape(h // ws, ws, w // ws, ws).transpose(0, 2, 1, 3).reshape(-1, ws * ws)
    diff = mw[:, None, :] - mw[:, :, None]
    return jnp.where(diff != 0, jnp.asarray(CONFIG['mask_value'], dtype), jnp.asarray(0.0, dtype))


def workload(x, qkv_w, qkv_b, proj_w, proj_b, table):
    b, h, w, c = x.shape
    ws, shift, nh = CONFIG['window_size'], CONFIG['shift_size'], CONFIG['num_heads']
    n = ws * ws
    hd = c // nh
    scale = hd ** -0.5

    shifted = jnp.roll(x, (-shift, -shift), axis=(1, 2))
    win = _partition(shifted, ws).reshape(-1, n, c)
    nw = win.shape[0]
    qkv = jnp.matmul(win, qkv_w, precision=_HI) + qkv_b
    qkv = qkv.reshape(nw, n, 3, nh, hd).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = jnp.matmul(q * scale, jnp.swapaxes(k, -1, -2), precision=_HI)

    bias = table[_rel_index(ws)].reshape(n, n, nh).transpose(2, 0, 1)
    attn = attn + bias[None]
    mask = _attn_mask(h, w, ws, shift, x.dtype)
    n_win = mask.shape[0]
    attn = attn.reshape(b, n_win, nh, n, n) + mask[None, :, None]
    attn = jax.nn.softmax(attn.reshape(-1, nh, n, n), axis=-1)

    out = jnp.matmul(attn, v, precision=_HI).transpose(0, 2, 1, 3).reshape(nw, n, c)
    out = jnp.matmul(out, proj_w, precision=_HI) + proj_b
    out = _reverse(out.reshape(-1, ws, ws, c), ws, b, h, w)
    return jnp.roll(out, (shift, shift), axis=(1, 2))
