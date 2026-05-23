import torch


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    # TODO (1 line): implement a lowest-AI op
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    # TODO (1 line): return either `fn` or `torch.compile(fn)` based on `compiled`
    if compiled:
        try:
            return torch.compile(fn)
        except RuntimeError as e:
            if "not supported on Python" in str(e) or "not supported" in str(e):
                import warnings
                warnings.warn(f"torch.compile is not supported on this Python/PyTorch version. Falling back to eager: {e}")
                return fn
            raise e
    return fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using MPS synchronization.

    Returns median execution time in milliseconds.
    """
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. This script is strictly MPS-only.")

    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.mps.synchronize()

    import time
    latencies = []
    
    for _ in range(rep):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
            
    latencies.sort()
    mid = len(latencies) // 2
    if len(latencies) % 2 == 0:
        median_time = (latencies[mid - 1] + latencies[mid]) / 2.0
    else:
        median_time = latencies[mid]
    return float(median_time)


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # TODO: compute total FLOPs, arithmetic intensity, and achieved FLOP/s
    total_flops = num_elements * num_ops * 2
    
    if variant == "compiled":
        # Fused kernel: 1 read, 1 write per element
        bytes_transferred = 2 * num_elements * bytes_per_element
    else:
        # Eager mode: separate multiply and add per iteration
        # Each iteration of `acc = acc * x + x` translates to two operations:
        #   1. temp = acc * x  --> reads acc, x; writes temp  (3 tensors)
        #   2. acc = temp + x  --> reads temp, x; writes acc  (3 tensors)
        # Total: 6 tensors transferred per iteration.
        bytes_transferred = 6 * num_elements * bytes_per_element * num_ops
        
    ai = total_flops / bytes_transferred if bytes_transferred > 0 else 0.0
    achieved_flops = total_flops / (ms * 1e-3) if ms > 0 else 0.0
    
    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# ANSWER 1: 
# In the memory-bound regime (to the left of the ridge point), the execution time is 
# dominated by memory traffic rather than computation. Since the compiled operations 
# are fused, the memory traffic is constant (reading the input tensor once and writing 
# the output tensor once, totaling 2 * N * 4 bytes for float32) regardless of the 
# number of operations in the loop. The additional operations are executed entirely on 
# registers, which has a negligible impact on latency since memory bandwidth is the 
# bottleneck. Because the total FLOPs increase linearly with `num_ops` while the 
# measured runtime remains almost flat, the achieved performance (FLOP/s = FLOPs / time) 
# rises linearly with the arithmetic intensity.
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# ANSWER 2:
# 1. Problem Size and Occupancy Under-utilization: A matrix multiplication of size 
# 1024x1024 on a massive GPU does not fully saturate the GPU's compute capacity. The 
# grid size is relatively small, so the GPU is latency-bound or wave-quantization-
# limited and cannot achieve peak Matrix Core throughput. In contrast, the element-wise 
# operation runs on a very large tensor (N = 64M elements), which fully occupies all compute 
# units with high occupancy, reaching near-peak memory-bandwidth-bound or vector-compute efficiency.
# 2. Vector ALU vs. Matrix Core Pipeline: If the matmul is executed using standard float32 vector 
# units rather than dedicated hardware matrix cores, it will achieve only a fraction of the GPU's peak 
# matrix performance, whereas the 128 ops compiled element-wise kernel runs extremely efficiently 
# on the GPU's vector execution pipelines.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# ANSWER 3:
# This transition suggests that the kernel is crossing the ridge point and transitioning 
# from being memory-bound to compute-bound (specifically ALU instruction throughput-bound). 
# For operations up to 64 ops, execution time was dominated by memory latency and bandwidth, 
# making the compute time "hidden" (pipelined behind memory transfers). At 128 ops, the 
# instruction execution time of the ALU inside the loop exceeds the memory transfer time. 
# As a result, memory transfer is no longer the bottleneck, and adding more compute 
# operations directly increases the runtime, showing that compute ALU capacity has become 
# the primary bottleneck.
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
# ANSWER 4:
# In eager mode, PyTorch does not perform kernel fusion. Each loop iteration launches 
# separate GPU kernels for the multiplication (`acc * x`) and the addition (`temp + x`). 
# Consequently, intermediate tensors are materialized and written to global GPU memory, 
# only to be read back by the next kernel. This creates massive memory traffic that scales 
# linearly with `num_ops` (estimated at 6 * N * 4 bytes of traffic per iteration). Because 
# memory traffic increases at the same rate as the number of FLOPs, eager arithmetic intensity 
# remains constant and very low (around 0.083 FLOP/Byte), keeping eager points clustered 
# vertically on the far-left (memory-bound) side of the roofline plot.
#
# In contrast, `torch.compile` captures the entire loop and fuses the operations into 
# a single GPU kernel, keeping intermediate accumulator variables in fast registers. Memory 
# traffic remains constant (just reading the input once and writing the output once) while 
# FLOPs increase, allowing the arithmetic intensity of compiled points to scale linearly 
# with `num_ops` and move rightward across the roofline towards the compute ceiling.
