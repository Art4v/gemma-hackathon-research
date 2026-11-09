"""[bf16 variant] TemporalBlock: causal-style end zero-pad on time axis -> Conv1d (valid, with bias) -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'TemporalBlock_16671',
    'batch': 16,
    'seq_len': 4096,
    'input_size': 512,
    'num_channels': 512,
    'kernel_size': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['input_size']), dtype)
    w = jax.random.normal(
        k_w, (CONFIG['num_channels'], CONFIG['input_size'], CONFIG['kernel_size']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['num_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # x: (B, L, C_in) is already NWC; w: (C_out, C_in, K) PyTorch OIW layout.
    k = w.shape[2]
    # ZeroPad2d((0,0,0,K-1)) on a 3D tensor pads the time axis at the end by K-1.
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(1,),
        padding=[(0, k - 1)],
        dimension_numbers=('NWC', 'OIW', 'NWC'),
        precision=lax.Precision.HIGHEST,
    )
    return jax.nn.relu(y + b)
