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
    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # TODO: time `rep` runs using CUDA events and return median latency (ms)
    latencies = []
    for _ in range(rep):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        fn(*args)
        end_event.record()
        
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
            
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
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# ANSWER 1:
# In the memory-bound regime (to the left of the ridge point), execution time is dominated
# by memory traffic rather than computation. Because the compiled operations are fused,
# the global memory traffic is constant: the input tensor is read once and the output tensor
# is written once, totaling 2 * N * 4 bytes (512 MB for float32 with N = 64M elements)
# regardless of `num_ops`. The loop's arithmetic operations execute entirely on registers,
# which has a negligible impact on latency since memory bandwidth is the bottleneck.
#
# Based on the L40S measurements:
#   - 1 ops (compiled): 0.844 ms, 0.16 TFLOP/s, AI = 0.25 FLOP/B
#   - 64 ops (compiled): 0.870 ms, 9.88 TFLOP/s, AI = 16.0 FLOP/B
# Although arithmetic intensity and FLOP count increased 64-fold, the runtime grew by less
# than 3% (0.844 ms to 0.870 ms). Since achieved performance is defined as FLOPs / time,
# and FLOPs scale linearly with `num_ops` while the runtime remains virtually flat,
# the achieved performance rises linearly with arithmetic intensity.
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# ANSWER 2:
# 1. SM Under-utilization and Wave Quantization: A matrix multiplication of size 1024x1024
#    is extremely small for massive modern GPUs like the H100 (132 SMs) or L40S (142 SMs).
#    It does not generate enough thread blocks to fully saturate and occupy the massive number
#    of SMs. This leads to severe under-utilization, where the GPU is latency-bound rather than
#    compute-throughput-bound. In contrast, the 128 ops compiled element-wise kernel runs on a
#    very large 1D tensor (64M elements), which guarantees maximum SM occupancy.
# 2. Kernel Launch and Host Overhead: For a small matrix (1024x1024), the execution time is
#    extremely short (~0.090 ms on L40S). At this scale, driver and kernel launch overheads
#    represent a significant portion of the total execution time, dragging down the achieved
#    FLOP/s. For the 128 ops compiled element-wise kernel, the runtime is 0.877 ms, allowing
#    the GPU to reach steady-state throughput and fully amortize launch overheads.
# 3. Vector ALU vs. Tensor Cores: Standard FP32 `torch.mm` utilizes standard FP32 vector ALUs
#    rather than Tensor Cores. Without Tensor Cores, the peak FP32 vector rate is a small fraction
#    of the GPU's peak tensor throughput, allowing a highly optimized Triton-compiled element-wise
#    vector ALU kernel to achieve comparable or higher TFLOP/s than a small non-tensor matmul.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# ANSWER 3:
# In general, a sudden, noticeable increase in runtime when scaling arithmetic intensity
# suggests that the kernel is crossing the "ridge point" of the roofline and transitioning
# from a memory-bandwidth-bound bottleneck to a compute-bound (ALU instruction throughput) bottleneck.
#
# Under this transition, the time required to execute the loop's arithmetic instructions
# on the ALU exceeds the memory transfer time. Thus, memory transfer is no longer the bottleneck,
# and adding more operations directly increases the runtime linearly.
#
# Hardware-specific observation on L40S vs. H100:
#   - On the H100, the ridge point is ~20 FLOP/Byte. Thus, 64 ops (AI = 16 FLOP/B) is near the ridge,
#     and 128 ops (AI = 32 FLOP/B) is fully compute-bound, resulting in a noticeable runtime jump.
#   - On the L40S, peak FP32 compute is 92 TFLOP/s and memory bandwidth is 0.86 TB/s, giving a
#     huge ridge point of 106.0 FLOP/Byte. This explains why the L40S runtime remained nearly
#     identical (0.870 ms for 64 ops vs. 0.877 ms for 128 ops) — the L40S is still strongly
#     memory-bound at 128 ops (AI = 32 FLOP/B), and the compute remains completely hidden.
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
# ANSWER 4:
# 1. Lack of Kernel Fusion: In eager mode, PyTorch evaluates every operation separately.
#    Each loop iteration `acc = acc * x + x` launches distinct CUDA kernels for the multiplication
#    and the addition. This forces intermediate tensors to be materialized and written back to
#    global GPU memory (HBM/GDDR), only to be read back immediately by the next kernel.
# 2. Arithmetic Intensity Bottleneck: Because eager mode transfers 6 tensors (4 reads and 2 writes)
#    per element per loop iteration, the memory traffic scales linearly with `num_ops` at a rate
#    of `6 * N * 4 * num_ops` bytes, while doing `2 * N * num_ops` FLOPs. This locks the eager arithmetic
#    intensity at a constant, extremely low value:
#      AI = (2 * N * num_ops) / (24 * N * num_ops) = 0.0833 FLOP/Byte.
#    Since both FLOPs and memory traffic grow at the same rate, the points remain clustered vertically
#    on the far-left memory-bound side of the plot.
# 3. Compiler Fusing: `torch.compile` captures the entire loop and uses Triton to generate a single,
#    fused GPU kernel. All intermediate accumulator updates are kept inside the GPU's fast registers,
#    so memory traffic is constant at `2 * N * 4` bytes (1 read and 1 write) while FLOPs grow linearly.
#    This allows the compiled arithmetic intensity to scale linearly with `num_ops` (AI = num_ops / 4),
#    moving the compiled points rightward across the roofline towards the peak compute ceiling.
#    Furthermore, eager mode suffers from severe kernel launch overhead due to launching 2 * num_ops
#    individual kernels, which scales runtime linearly and keeps performance extremely low (~0.05 TFLOP/s).
