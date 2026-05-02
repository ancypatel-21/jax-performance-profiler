🚀 JAX Performance Profiler & Computation Tracer
📌 Overview

This project demonstrates how JAX optimizes numerical computations using JIT (Just-In-Time) compilation and computation graph tracing (JAXPR). It compares execution performance between standard and optimized runs.

🎯 Features
Trace computation graphs using jax.make_jaxpr
Benchmark performance with and without jax.jit
Measure speedup on large vectorized inputs (1M+ data points)
Modular design (tracing + benchmarking)
🏗️ Project Structure
jax-profiler/
│── main.py
│── tracer.py
│── benchmark.py
│── README.md
⚙️ How It Works
1. Tracing (JAXPR)

JAX converts Python functions into a low-level representation:

jax.make_jaxpr(function)
2. JIT Compilation
jax.jit(function)
First run → includes compilation time
Subsequent runs → faster execution
3. Benchmarking

Compares:

Normal execution
JIT (compile + run)
JIT (optimized run)
▶️ Run the Project
python3 main.py
📈 Sample Output
Without JIT: 0.27 sec
JIT (compile+run): 0.14 sec
JIT (fast run): 0.03 sec
Speedup: 8.4x
🧠 Key Learnings
JAX uses XLA compilation for optimization
JAXPR helps visualize execution flow
JIT significantly improves performance for large workloads
🛠️ Tech Stack
Python
JAX
Numerical Computing
