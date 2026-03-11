# Teacher Feedback Response

## Feedback

The course requires low-level GPU programming with custom CUDA/HIP kernels. PyTorch-based kernels are not acceptable for the main implementation.

## Changes made

1. Replaced the default runtime path with a custom CUDA implementation in `cuda_vsr/`.
2. Added custom kernels for:
   - temporal fusion
   - bilinear resize
   - sharpen post-process
3. Updated the entrypoint `main.sh` to run `scripts/run_cuda_vsr_mahti.sh`.
4. Updated SLURM scripts to profile and run the CUDA binary directly.
5. Updated project documentation with Mahti-specific build/run instructions.

## How the new solution aligns with course goals

- Kernel code is explicitly written in CUDA C++ (`cuda_vsr/src/kernels.cu`).
- Performance analysis can be done with Nsight Compute and Nsight Systems scripts.
- Bottleneck reporting is exported to JSON metrics for each run.
