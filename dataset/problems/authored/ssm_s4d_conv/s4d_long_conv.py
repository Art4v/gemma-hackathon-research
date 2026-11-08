"""S4D layer causal long convolution via ZOH kernel generation and direct depthwise conv."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 's4d_long_conv',
    'batch_size': 32,
    'd_model': 1024,
    'seq_len': 1024,
    'd_state': 64,
    'dt_min': -4.0,
    'dt_max': -2.0,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_u, k_dt, k_a, k_b, k_c, k_d = jax.random.split(key, 6)
    
    B = CONFIG['batch_size']
    H = CONFIG['d_model']
    L = CONFIG['seq_len']
    N = CONFIG['d_state']
    
    u = jax.random.normal(k_u, (B, H, L), dtype=dtype)
    
    log_dt = jax.random.uniform(
        k_dt, (H,), dtype=jnp.float32,
        minval=CONFIG['dt_min'], maxval=CONFIG['dt_max']
    ).astype(dtype)
    log_A_real = jax.random.uniform(
        k_a, (H, N), dtype=jnp.float32,
        minval=-2.0, maxval=0.0
    ).astype(dtype)
    B_proj = jax.random.normal(k_b, (H, N), dtype=dtype) * 0.02
    C_proj = jax.random.normal(k_c, (H, N), dtype=dtype) * 0.02
    D = jnp.ones((H,), dtype=dtype)
    
    return u, log_dt, log_A_real, B_proj, C_proj, D

def workload(u, log_dt, log_A_real, B_proj, C_proj, D):
    B, H, L = u.shape
    
    u_f32 = u.astype(jnp.float32)
    dt = jnp.exp(log_dt.astype(jnp.float32))
    A = -jnp.exp(log_A_real.astype(jnp.float32))
    B_f32 = B_proj.astype(jnp.float32)
    C_f32 = C_proj.astype(jnp.float32)
    D_f32 = D.astype(jnp.float32)
    
    dtA = dt[:, None] * A
    B_tilde = B_f32 * C_f32 * (jnp.expm1(dtA) / A)
    
    l_idx = jnp.arange(L, dtype=jnp.float32)
    vandermonde = jnp.exp(dtA[:, :, None] * l_idx[None, None, :])
    K = jnp.einsum('hn,hnl->hl', B_tilde, vandermonde)
    
    K_flipped = jnp.flip(K, axis=-1)[:, None, :]
    
    y_conv = lax.conv_general_dilated(
        lhs=u_f32,
        rhs=K_flipped,
        window_strides=(1,),
        padding=((L - 1, 0),),
        feature_group_count=H,
        dimension_numbers=('NCH', 'OIH', 'NCH'),
        preferred_element_type=jnp.float32,
    )
    
    y = y_conv + u_f32 * D_f32[None, :, None]
    return y.astype(u.dtype)
