#include "kernels.cuh"

#include <math.h>
#include <stdexcept>
#include <string>

namespace {

__device__ __forceinline__ float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

__global__ void u8_to_f32_kernel(const unsigned char* in, float* out, int numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = static_cast<float>(in[idx]) * (1.0f / 255.0f);
    }
}

__global__ void f32_to_u8_kernel(const float* in, unsigned char* out, int numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        const float v = clamp01(in[idx]) * 255.0f;
        out[idx] = static_cast<unsigned char>(v + 0.5f);
    }
}

__global__ void temporal_fusion_kernel(
    const float* prev,
    const float* curr,
    const float* next,
    float* fused,
    int numel,
    float w_prev,
    float w_curr,
    float w_next
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        fused[idx] = w_prev * prev[idx] + w_curr * curr[idx] + w_next * next[idx];
    }
}

__global__ void resize_bilinear_rgb_kernel(
    const float* in,
    int in_w,
    int in_h,
    float* out,
    int out_w,
    int out_h
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) {
        return;
    }

    const float src_x = ((static_cast<float>(x) + 0.5f) * static_cast<float>(in_w) / static_cast<float>(out_w)) - 0.5f;
    const float src_y = ((static_cast<float>(y) + 0.5f) * static_cast<float>(in_h) / static_cast<float>(out_h)) - 0.5f;

    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    const float ax = src_x - floorf(src_x);
    const float ay = src_y - floorf(src_y);

    if (x0 < 0) {
        x0 = 0;
    }
    if (y0 < 0) {
        y0 = 0;
    }
    const int x1 = (x0 + 1 < in_w) ? x0 + 1 : x0;
    const int y1 = (y0 + 1 < in_h) ? y0 + 1 : y0;

    const int out_base = (y * out_w + x) * 3;
    const int p00 = (y0 * in_w + x0) * 3;
    const int p10 = (y0 * in_w + x1) * 3;
    const int p01 = (y1 * in_w + x0) * 3;
    const int p11 = (y1 * in_w + x1) * 3;

    for (int c = 0; c < 3; ++c) {
        const float v00 = in[p00 + c];
        const float v10 = in[p10 + c];
        const float v01 = in[p01 + c];
        const float v11 = in[p11 + c];
        const float top = v00 + (v10 - v00) * ax;
        const float bottom = v01 + (v11 - v01) * ax;
        out[out_base + c] = top + (bottom - top) * ay;
    }
}

__global__ void sharpen_rgb_kernel(
    const float* in,
    float* out,
    int w,
    int h,
    float amount
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    const int xl = (x > 0) ? x - 1 : x;
    const int xr = (x + 1 < w) ? x + 1 : x;
    const int yu = (y > 0) ? y - 1 : y;
    const int yd = (y + 1 < h) ? y + 1 : y;

    const int center = (y * w + x) * 3;
    const int left = (y * w + xl) * 3;
    const int right = (y * w + xr) * 3;
    const int up = (yu * w + x) * 3;
    const int down = (yd * w + x) * 3;

    for (int c = 0; c < 3; ++c) {
        const float c0 = in[center + c];
        const float blur = (4.0f * c0 + in[left + c] + in[right + c] + in[up + c] + in[down + c]) * (1.0f / 8.0f);
        const float sharpened = c0 + amount * (c0 - blur);
        out[center + c] = clamp01(sharpened);
    }
}

}  // namespace

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

void launch_u8_to_f32(const unsigned char* in, float* out, int numel, cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    u8_to_f32_kernel<<<blocks, threads, 0, stream>>>(in, out, numel);
    check_cuda(cudaGetLastError(), "launch_u8_to_f32");
}

void launch_f32_to_u8(const float* in, unsigned char* out, int numel, cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    f32_to_u8_kernel<<<blocks, threads, 0, stream>>>(in, out, numel);
    check_cuda(cudaGetLastError(), "launch_f32_to_u8");
}

void launch_temporal_fusion(
    const float* prev,
    const float* curr,
    const float* next,
    float* fused,
    int numel,
    float w_prev,
    float w_curr,
    float w_next,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    temporal_fusion_kernel<<<blocks, threads, 0, stream>>>(prev, curr, next, fused, numel, w_prev, w_curr, w_next);
    check_cuda(cudaGetLastError(), "launch_temporal_fusion");
}

void launch_resize_bilinear_rgb(
    const float* in,
    int in_w,
    int in_h,
    float* out,
    int out_w,
    int out_h,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
    resize_bilinear_rgb_kernel<<<grid, block, 0, stream>>>(in, in_w, in_h, out, out_w, out_h);
    check_cuda(cudaGetLastError(), "launch_resize_bilinear_rgb");
}

void launch_sharpen_rgb(const float* in, float* out, int w, int h, float amount, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    sharpen_rgb_kernel<<<grid, block, 0, stream>>>(in, out, w, h, amount);
    check_cuda(cudaGetLastError(), "launch_sharpen_rgb");
}
