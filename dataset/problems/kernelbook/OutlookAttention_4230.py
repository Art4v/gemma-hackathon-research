"""[bf16 variant] OutlookAttention: v-linear -> unfold -> avgpool -> attn-linear -> scaled softmax -> attn@v -> fold -> proj-linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'OutlookAttention_4230',
    'batch': 8,
    'height': 128,
    'width': 128,
    'dim': 128,
    'num_heads': 1,
    'kernel_size': 3,
    'padding': 1,
    'stride': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    B, H, W, C = CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['dim']
    k, nh = CONFIG['kernel_size'], CONFIG['num_heads']
    a_out = k ** 4 * nh
    keys = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(keys[0], (B, H, W, C), dtype)
    w_v = jax.random.normal(keys[1], (C, C), dtype) * 0.02
    w_attn = jax.random.normal(keys[2], (C, a_out), dtype) * 0.02
    b_attn = jax.random.normal(keys[3], (a_out,), dtype) * 0.02
    w_proj = jax.random.normal(keys[4], (C, C), dtype) * 0.02
    b_proj = jax.random.normal(keys[5], (C,), dtype) * 0.02
    return x, w_v, w_attn, b_attn, w_proj, b_proj


def _avg_pool_ceil(x, s):
    # AvgPool2d(kernel=s, stride=s, ceil_mode=True, padding=0) on NHWC: edge windows
    # divide by their in-bounds element count, as PyTorch does.
    _, H, W, _ = x.shape
    h, w = -(-H // s), -(-W // s)
    pad = ((0, 0), (0, h * s - H), (0, w * s - W), (0, 0))
    win = (1, s, s, 1)
    total = lax.reduce_window(x, 0.0, lax.add, win, win, pad)
    count = lax.reduce_window(jnp.ones((1, H, W, 1), x.dtype), 0.0, lax.add, win, win, pad)
    return total / count


def workload(x, w_v, w_attn, b_attn, w_proj, b_proj):
    B, H, W, C = x.shape
    nh, k = CONFIG['num_heads'], CONFIG['kernel_size']
    p, s = CONFIG['padding'], CONFIG['stride']
    hd = C // nh
    K2 = k * k
    h, w = -(-H // s), -(-W // s)
    scale = hd ** -0.5

    # Unfold v into (B, h, w, K2, C); channel c = head * hd + d.
    v = x @ w_v
    vp = jnp.pad(v, ((0, 0), (p, p), (p, p), (0, 0)))
    span_h, span_w = s * (h - 1) + 1, s * (w - 1) + 1
    patches = jnp.stack(
        [vp[:, ki:ki + span_h:s, kj:kj + span_w:s, :] for ki in range(k) for kj in range(k)],
        axis=3)
    v_heads = patches.reshape(B, h * w, K2, nh, hd).transpose(0, 3, 1, 2, 4)  # (B, nh, hw, K2, hd)

    pooled = _avg_pool_ceil(x, s)
    attn = (pooled @ w_attn + b_attn).reshape(B, h * w, nh, K2, K2).transpose(0, 2, 1, 3, 4)
    attn = jax.nn.softmax(attn * scale, axis=-1)

    out = jnp.matmul(attn, v_heads)  # (B, nh, hw, K2, hd)
    out = out.transpose(0, 2, 3, 1, 4).reshape(B, h, w, K2, C)

    # Fold: scatter-add each kernel offset back onto the padded canvas, then crop.
    canvas = jnp.zeros((B, H + 2 * p, W + 2 * p, C), x.dtype)
    for ki in range(k):
        for kj in range(k):
            canvas = canvas.at[:, ki:ki + span_h:s, kj:kj + span_w:s, :].add(out[:, :, :, ki * k + kj, :])
    folded = canvas[:, p:p + H, p:p + W, :]

    return folded @ w_proj + b_proj
