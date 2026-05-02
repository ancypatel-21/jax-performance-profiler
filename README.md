📌 Overview

This project demonstrates how JAX optimizes numerical computations using JIT (Just-In-Time) compilation and computation graph tracing (JAXPR). It analyzes how functions are executed internally and compares performance between normal execution and optimized execution.

🎯 Features
🔍 Trace computation graphs using jax.make_jaxpr
⚡ Benchmark performance with and without jax.jit
📊 Measure real speedup on large vectorized inputs (1M+ data points)
🧩 Modular structure (tracing + benchmarking)
🏗️ Project Structure

jax-profiler/
│── main.py
│── tracer.py
│── benchmark.py
│── README.md

⚙️ How It Works
1. Tracing (JAXPR)

JAX converts Python functions into a low-level representation using:
jax.make_jaxpr(function)

This shows operations like sin, cos, add, and multiply step-by-step.

2. JIT Compilation

Using:
jax.jit(function)

First run → includes compilation time
Subsequent runs → significantly faster execution
3. Benchmarking

The project compares:

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
JAX uses XLA compilation for performance optimization
Computation graphs (JAXPR) help understand execution flow
JIT significantly improves performance for large workloads
🛠️ Tech Stack
Python
JAX
Numerical Computing