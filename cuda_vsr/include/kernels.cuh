#pragma once

#include <cuda_runtime.h>

void check_cuda(cudaError_t err, const char* msg);

void launch_u8_to_f32(
    const unsigned char* in,
    float* out,
    int numel,
    cudaStream_t stream = nullptr
);

void launch_f32_to_u8(
    const float* in,
    unsigned char* out,
    int numel,
    cudaStream_t stream = nullptr
);

void launch_temporal_fusion(
    const float* prev,
    const float* curr,
    const float* next,
    float* fused,
    int numel,
    float w_prev,
    float w_curr,
    float w_next,
    cudaStream_t stream = nullptr
);

void launch_resize_bilinear_rgb(
    const float* in,
    int in_w,
    int in_h,
    float* out,
    int out_w,
    int out_h,
    cudaStream_t stream = nullptr
);

void launch_sharpen_rgb(
    const float* in,
    float* out,
    int w,
    int h,
    float amount,
    cudaStream_t stream = nullptr
);
