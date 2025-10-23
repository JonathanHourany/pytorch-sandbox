# Profiling Tests

When training neural networks, we aim to maximize GPU utilization since CUDA cores are
optimized for large-scale parallel computation. In practice, utilization often falls
short due to bottlenecks elsewhere in the training pipeline. By profiling our code, we
can identify where time is being spent and uncover opportunities for optimization.

This repository highlights several common causes of low GPU utilization or memory
inefficiencies and demonstrates how they appear in profiling traces.

## Data Bottlenecks

This section shows profiling outputs where the GPU is underutilized because it is
waiting on the data pipeline.

- **Too few DataLoader workers**: If the `DataLoader` cannot supply batches fast
  enough, the GPU idles while waiting for data to be prepared and transferred. In
  profiling traces, this appears as large gaps of idle time between GPU kernel
  executions.

- **Small batch sizes**: Very small batches do not fully saturate the GPU, so kernels
  finish execution quickly. The GPU then stalls while waiting for the next batch. In
  profiling traces, this appears as many short-lived kernels (often < 50–100 µs) with
  low overall utilization.

## Mixed Precision Tensors

The `float32` (FP32) data type uses 23 bits for the significand (mantissa), which gives
high precision but is often unnecessary for deep learning workloads. In many cases,
we can achieve significant speedups and memory savings by using `bfloat16` (BF16).

Like FP32, BF16 uses 8 bits for the exponent, so it maintains the same dynamic range
(i.e., it can represent equally large and small numbers). However, it only allocates
7 bits to the significand, which reduces numerical precision. This tradeoff allows
for major efficiency gains:

- **Memory Bandwidth**: Twice as many values can be transferred between host and
  device compared to FP32, reducing data movement bottlenecks.
- **Data Processing**: Registers can pack twice as many BF16 values, and modern
  accelerators (e.g., NVIDIA Tensor Cores, Google TPUs) can perform up to 4× more
  16-bit operations than 32-bit ones.

Together, these improvements can dramatically reduce training time. In practice,
enabling **mixed precision training** (FP16 or BF16 for most operations, FP32 for
critical ones like loss scaling) often cuts training time significantly. Profiler
output for the script in this folder shows total CUDA time reduced by ~50% when using
mixed precision compared to full precision.precision.

## Activation Recomputation

During training, intermediate activations are required to compute gradients in the
backward pass. However, depending on the model architecture, batch size, and sequence
length, storing all activations can consume a large amount of GPU memory. This often
forces smaller batch sizes or limits model depth.

In memory profiling, this appears as a steady increase in memory usage during the
forward pass, followed by memory being released during the backward pass as gradients
are computed and the corresponding activations are freed.

A common technique to reduce memory usage is **activation recomputation**, also known
as **gradient checkpointing**. Instead of storing every intermediate activation, the
model saves only selected **checkpoints** (the outputs of certain layers or blocks).
During the backward pass, the missing activations are recomputed by re-running parts
of the forward pass from the nearest checkpoint.

This trades off additional computation time for significantly lower memory usage,
often allowing for much larger models or batch sizes to fit into GPU memory.

## Kernel Launch Overhead

Launching many small kernels in rapid succession can introduce overhead that reduces
throughput. This often occurs when operations are not fused, or when excessive tensor
manipulations happen in Python. Profiling traces will show a large number of very
short kernels interspersed with idle gaps. At the moment, there isn't a demo of this in 
this repo.

## Synchronization Issues

Certain operations (e.g., calling `.item()` on a GPU tensor, or synchronizing across
multiple devices) force the CPU and GPU to wait on each other. These blocking calls
can drastically reduce utilization. In profiling traces, you may see frequent stalls
where the GPU waits for the host to catch up. At the moment, there isn't a demo of this 
in this repo.

---

By studying these bottlenecks in profiling traces, we can better understand where our
training pipelines spend time, how efficiently we are using the available resources
and apply optimizations.
ß