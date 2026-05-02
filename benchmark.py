import time
import jax

def benchmark(func, jit_func, x):
    # Without JIT
    start = time.time()
    func(x).block_until_ready()
    t1 = time.time() - start

    # JIT compile + run
    start = time.time()
    jit_func(x).block_until_ready()
    t2 = time.time() - start

    # JIT fast run
    start = time.time()
    jit_func(x).block_until_ready()
    t3 = time.time() - start

    print("\n=== Performance ===")
    print(f"Without JIT: {t1:.4f} sec")
    print(f"JIT (compile+run): {t2:.4f} sec")
    print(f"JIT (fast run): {t3:.4f} sec")
    print(f"Speedup: {t1/t3:.2f}x")