"""[bf16 variant] Time-varying causal FIR filter: y[b,n,c] = sum_k signal[b,n-k,c] * coef[b,n,k]."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'TimeVarFIRFilter_14091',
    'batch': 8,
    'signal_length': 32768,
    'channels': 128,
    'filter_order': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    k_sig, k_coef = jax.random.split(jax.random.key(0))
    b, l = CONFIG['batch'], CONFIG['signal_length']
    signal = jax.random.uniform(k_sig, (b, l, CONFIG['channels']), dtype)
    f_coef = jax.random.uniform(k_coef, (b, l, CONFIG['filter_order']), dtype)
    return signal, f_coef


def workload(signal, f_coef):
    b, l, c = signal.shape
    order_k = f_coef.shape[-1]
    # Front zero-padding by K-1 equals torch's end-pad + roll: wrapped values are always zeros.
    padded = jnp.pad(signal, ((0, 0), (order_k - 1, 0), (0, 0)))

    def body(k, y):
        shifted = lax.dynamic_slice(padded, (0, order_k - 1 - k, 0), (b, l, c))
        coef_k = lax.dynamic_slice(f_coef, (0, 0, k), (b, l, 1))
        return y + shifted * coef_k

    return lax.fori_loop(0, order_k, body, jnp.zeros_like(signal))
