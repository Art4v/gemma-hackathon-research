"""[bf16 variant] TAE decoder: nearest Upsample1d -> ConvTranspose1d with bias -> crop to pooling -> flatten."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'TAE_decoder_8212',
    'batch': 1024,
    'n_hidden': 1024,
    'input_length': 4,
    'pooling': 8,
    'kernel_size': 10,
    'stride': 1,
    'padding': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_features, k_weight, k_bias = jax.random.split(key, 3)
    features = jax.random.uniform(
        k_features,
        (CONFIG['batch'], CONFIG['n_hidden'], CONFIG['input_length']),
        dtype,
    )
    # PyTorch ConvTranspose1d layout: (in_channels, out_channels, kernel_size)
    weight = jax.random.normal(
        k_weight,
        (CONFIG['n_hidden'], CONFIG['n_hidden'], CONFIG['kernel_size']),
        dtype,
    ) * 0.02
    bias = jax.random.normal(k_bias, (CONFIG['n_hidden'],), dtype) * 0.02
    return features, weight, bias


def workload(features, weight, bias):
    pooling = CONFIG['pooling']
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    kernel_size = weight.shape[2]

    # Nearest-neighbour upsample along the length axis (NCW).
    input_length = features.shape[2]
    indices = (jnp.arange(pooling) * input_length) // pooling
    upsampled = jnp.take(features, indices, axis=2)

    # Transposed convolution expressed as a regular conv with a reversed kernel.
    x = jnp.transpose(upsampled, (0, 2, 1))  # NCW -> NWC
    kernel = jnp.flip(jnp.transpose(weight, (2, 0, 1)), axis=0)  # (in, out, k) -> WIO
    pad = kernel_size - 1 - padding
    y = lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(stride,),
        padding=[(pad, pad)],
        lhs_dilation=(stride,),
        rhs_dilation=(1,),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
    )
    y = y + bias

    # Crop and flatten in PyTorch's NCW element order.
    y = jnp.transpose(y, (0, 2, 1))  # NWC -> NCW
    y = y[:, :, :pooling]
    return y.reshape(y.shape[0], -1)
