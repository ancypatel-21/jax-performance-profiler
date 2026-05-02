import jax
import jax.numpy as jnp
from tracer import trace_function
from benchmark import benchmark

# Function
def compute_function(x):
    for _ in range(50):
        x = jnp.sin(x) * jnp.cos(x) + x**2
    return x

jit_func = jax.jit(compute_function)

# Small input (for tracing)
x_small = jnp.array([1.0, 2.0, 3.0])
trace_function(compute_function, x_small)

# Large input (for benchmarking)
x_large = jnp.linspace(0, 10, 1_000_000)
benchmark(compute_function, jit_func, x_large)