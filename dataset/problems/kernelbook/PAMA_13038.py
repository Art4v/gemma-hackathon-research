"""[bf16 variant] PAMA: reflect-pad conv3x3 + ReLU6 -> mean-variance-norm attention (1x1 convs, softmax, bmm) -> conv3x3 -> gated multi-scale fuse."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'PAMA_13038',
    'batch': 8,
    'channels': 256,
    'height': 64,
    'width': 64,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    B, C, H, W = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    Ch = C // 2
    keys = jax.random.split(jax.random.key(0), 26)
    fc = jax.random.uniform(keys[0], (B, C, H, W), dtype)
    fs = jax.random.uniform(keys[1], (B, C, H, W), dtype)
    shapes = [
        (3, 3, C, C), (C,),      # conv_in (HWIO)
        (C, Ch), (Ch,),          # attn f
        (C, Ch), (Ch,),          # attn g
        (C, Ch), (Ch,),          # attn h
        (Ch, C), (C,),           # attn out_conv
        (3, 3, C, C), (C,),      # conv_out (HWIO)
        (2 * C, C), (C,),        # fuse proj1
        (C, C), (C,),            # fuse proj2
        (C, C), (C,),            # fuse proj3
        (C, 1), (1,),            # fuse1x
        (3, 3, C, 1), (1,),      # fuse3x (HWIO)
        (5, 5, C, 1), (1,),      # fuse5x (HWIO)
    ]
    params = tuple(
        jax.random.normal(k, s, dtype) * 0.02 for k, s in zip(keys[2:], shapes)
    )
    return (fc, fs) + params


def _relu6(x):
    return jnp.clip(x, 0.0, 6.0)


def _conv_reflect(x, w, b):
    # x NHWC, w HWIO; reflection pad keeps spatial size like nn.ReflectionPad2d + VALID conv.
    p = w.shape[0] // 2
    xp = jnp.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='reflect')
    y = lax.conv_general_dilated(
        xp, w, (1, 1), 'VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def _mvn(x):
    # Unbiased variance over spatial dims, matching torch .var default.
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.var(x, axis=(1, 2), keepdims=True, ddof=1) + CONFIG['eps']
    return (x - mean) / jnp.sqrt(var)


def workload(fc, fs,
             conv_in_w, conv_in_b,
             f_w, f_b, g_w, g_b, h_w, h_b, attn_out_w, attn_out_b,
             conv_out_w, conv_out_b,
             proj1_w, proj1_b, proj2_w, proj2_b, proj3_w, proj3_b,
             fuse1_w, fuse1_b, fuse3_w, fuse3_b, fuse5_w, fuse5_b):
    B, C, H, W = fc.shape
    N = H * W
    fc = jnp.transpose(fc, (0, 2, 3, 1))
    fs = jnp.transpose(fs, (0, 2, 3, 1))

    fc = _relu6(_conv_reflect(fc, conv_in_w, conv_in_b))
    fs = _relu6(_conv_reflect(fs, conv_in_w, conv_in_b))

    # Attention unit (tokens = spatial positions).
    f = _relu6(_mvn(fc).reshape(B, N, C) @ f_w + f_b)
    g = _relu6(_mvn(fs).reshape(B, N, C) @ g_w + g_b)
    h = _relu6(fs.reshape(B, N, C) @ h_w + h_b)
    attn = jax.nn.softmax(jnp.einsum('bic,bjc->bij', f, g), axis=-1)
    fcs = jnp.einsum('bij,bjc->bic', attn, h)
    fcs = _relu6(fcs @ attn_out_w + attn_out_b).reshape(B, H, W, C)

    fcs = _relu6(_conv_reflect(fcs, conv_out_w, conv_out_b))

    # Fuse unit.
    fcat = jnp.concatenate((fc, fcs), axis=-1) @ proj1_w + proj1_b
    f1 = fc @ proj2_w + proj2_b
    f2 = fcs @ proj3_w + proj3_b
    fusion1 = jax.nn.sigmoid(fcat @ fuse1_w + fuse1_b)
    fusion3 = jax.nn.sigmoid(_conv_reflect(fcat, fuse3_w, fuse3_b))
    fusion5 = jax.nn.sigmoid(_conv_reflect(fcat, fuse5_w, fuse5_b))
    fusion = (fusion1 + fusion3 + fusion5) / 3
    out = jnp.clip(fusion, 0.0, 1.0) * f1 + jnp.clip(1 - fusion, 0.0, 1.0) * f2
    return jnp.transpose(out, (0, 3, 1, 2))
