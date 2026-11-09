"""[bf16 variant] UpConcat2d: ConvTranspose2d(k=2,s=2) -> ReLU -> pad/center-crop skip -> channel concat."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'UpConcat2d_7560',
    'batch': 8,
    'in_channels_conv': 128,
    'out_channels_conv': 128,
    'enc_channels': 128,
    'height': 128,
    'width': 128,
    'enc_height': 128,
    'enc_width': 128,
    'kernel_size': 2,
    'stride': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    x_down = jax.random.uniform(k1, (c['batch'], c['in_channels_conv'], c['height'], c['width']), dtype)
    x_enc = jax.random.uniform(k2, (c['batch'], c['enc_channels'], c['enc_height'], c['enc_width']), dtype)
    w = jax.random.normal(k3, (c['in_channels_conv'], c['out_channels_conv'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(k4, (c['out_channels_conv'],), dtype) * 0.02
    return x_down, x_enc, w, b


def workload(x_down, x_enc, w, b):
    n, _, h, wd = x_down.shape
    cout, k = w.shape[1], w.shape[2]
    # stride == kernel, so the transposed conv has no overlap: each input pixel writes its own k x k block.
    up = jnp.einsum('nchw,coab->nohawb', x_down, w).reshape(n, cout, h * k, wd * k)
    up = jax.nn.relu(up + b[None, :, None, None])

    uh, uw = up.shape[-2], up.shape[-1]
    if uw > x_enc.shape[-1]:
        diff = uw - x_enc.shape[-1]
        p = diff // 2 + (diff % 2)
        x_enc = jnp.pad(x_enc, ((0, 0), (0, 0), (p, p), (p, p)))
    sh = (x_enc.shape[-2] - uh) // 2
    sw = (x_enc.shape[-1] - uw) // 2
    crop = x_enc[:, :, sh:sh + uh, sw:sw + uw]
    return jnp.concatenate([up, crop], axis=1)
