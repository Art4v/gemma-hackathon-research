"""NonLocalBlock: 1x1 convs (phi/theta/g) -> quirky view -> theta@phi -> softmax(dim=1) -> @g -> 1x1 conv_mask -> + x."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NonLocalBlock_8363',
    'batch': 8,
    'channel': 512,
    'height': 64,
    'width': 64,
}


def create_inputs(dtype=jnp.float32):
    b, c, h, w = CONFIG['batch'], CONFIG['channel'], CONFIG['height'], CONFIG['width']
    ic = c // 2
    kx, kphi, ktheta, kg, kmask = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (b, c, h, w), dtype)
    w_phi = jax.random.normal(kphi, (1, 1, c, ic), dtype) * 0.02
    w_theta = jax.random.normal(ktheta, (1, 1, c, ic), dtype) * 0.02
    w_g = jax.random.normal(kg, (1, 1, c, ic), dtype) * 0.02
    w_mask = jax.random.normal(kmask, (1, 1, ic, c), dtype) * 0.02
    return x, w_phi, w_theta, w_g, w_mask


def _conv1x1_nchw(x, w_hwio):
    y = lax.conv_general_dilated(
        jnp.transpose(x, (0, 2, 3, 1)), w_hwio,
        window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return jnp.transpose(y, (0, 3, 1, 2))


def workload(x, w_phi, w_theta, w_g, w_mask):
    b, c, h, w = x.shape
    ic = w_phi.shape[-1]
    # torch views the (b, c/2, h, w) conv output as (b, c, h*w/2); keep that exact reshape.
    x_phi = _conv1x1_nchw(x, w_phi).reshape(b, c, -1)
    x_theta = jnp.transpose(_conv1x1_nchw(x, w_theta).reshape(b, c, -1), (0, 2, 1))
    x_g = jnp.transpose(_conv1x1_nchw(x, w_g).reshape(b, c, -1), (0, 2, 1))
    scores = jnp.matmul(x_theta, x_phi)
    scores = jax.nn.softmax(scores, axis=1)
    y = jnp.matmul(scores, x_g)
    y = jnp.transpose(y, (0, 2, 1)).reshape(b, ic, h, w)
    return _conv1x1_nchw(y, w_mask) + x
