# Profiling Tests

When training neural networks, we aim to maximize GPU utilization since CUDA cores are
optimized for large-scale parallel computation. In practice, utilization often falls
short due to bottlenecks elsewhere in the training pipeline. By profiling our code, we
can identify where time is being spent and uncover opportunities for optimization.

This repository highlights several common causes of low GPU utilization and
demonstrates how they appear in profiling traces.

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

## Backward Pass Imbalance

...

## Kernel Launch Overhead

Launching many small kernels in rapid succession can introduce overhead that reduces
throughput. This often occurs when operations are not fused, or when excessive tensor
manipulations happen in Python. Profiling traces will show a large number of very
short kernels interspersed with idle gaps.

## Synchronization Issues

Certain operations (e.g., calling `.item()` on a GPU tensor, or synchronizing across
multiple devices) force the CPU and GPU to wait on each other. These blocking calls
can drastically reduce utilization. In profiling traces, you may see frequent stalls
where the GPU waits for the host to catch up.

---

By studying these bottlenecks in profiling traces, we can better understand where our
training pipelines spend time and apply optimizations such as increasing batch size,
tuning the number of `DataLoader` workers, moving preprocessing off the critical path,
or using fused kernels.
