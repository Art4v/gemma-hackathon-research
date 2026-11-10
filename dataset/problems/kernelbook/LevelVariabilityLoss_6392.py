"""[bf16 variant] LevelVariabilityLoss: log -> first diff along time -> second diff -> squared mean * penalty."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LevelVariabilityLoss_6392',
    'batch': 8,
    'n_time': 512,
    'channels': 128,
    'features': 128,
    'level_variability_penalty': 4.0,
    'min_level': 1e-3,
    'max_level': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['n_time'], CONFIG['channels'], CONFIG['features'])
    levels = jax.random.uniform(
        key, shape, dtype, minval=CONFIG['min_level'], maxval=CONFIG['max_level']
    )
    return (levels,)


def workload(levels):
    log_levels = jnp.log(levels)
    level_prev = log_levels[:, :-1]
    level_next = log_levels[:, 1:]
    log_diff_of_levels = level_prev - level_next
    log_diff_prev = log_diff_of_levels[:, :-1]
    log_diff_next = log_diff_of_levels[:, 1:]
    diff = log_diff_prev - log_diff_next
    return jnp.mean(diff ** 2) * CONFIG['level_variability_penalty']
