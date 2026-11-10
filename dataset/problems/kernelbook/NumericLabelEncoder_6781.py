"""NumericLabelEncoder: flatten -> broadcast equality against class table -> int cast -> argmax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NumericLabelEncoder_6781',
    'input_shape': (16, 128, 128, 128),
    'classes': 8,
    'value_range': 16,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_check = jax.random.split(key)
    x = jnp.floor(
        jax.random.uniform(k_x, CONFIG['input_shape'], dtype) * CONFIG['value_range']
    )
    check_tensor = jax.random.randint(
        k_check, (CONFIG['classes'],), 0, CONFIG['value_range'], dtype=jnp.int32
    )
    return x, check_tensor


def workload(x, check_tensor):
    x = x.reshape(-1, 1)
    matches = jnp.equal(x, check_tensor.astype(x.dtype)).astype(jnp.int32)
    return jnp.argmax(matches, axis=1)
