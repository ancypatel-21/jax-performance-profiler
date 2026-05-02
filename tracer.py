import jax

def trace_function(func, x):
    print("=== JAXPR (Computation Graph) ===")
    print(jax.make_jaxpr(func)(x))