#include "kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

struct Args {
    std::string input_dir;
    std::string output_dir;
    std::string backend = "cuda";
    std::string metrics_path;
    int target_height = 0;
    float w_prev = 0.2f;
    float w_curr = 0.6f;
    float w_next = 0.2f;
    float sharpen = 0.25f;
};

struct Metrics {
    int frames = 0;
    double load_ms = 0.0;
    double h2d_ms = 0.0;
    double d2h_ms = 0.0;
    double u8_to_f32_ms = 0.0;
    double temporal_ms = 0.0;
    double resize_ms = 0.0;
    double sharpen_ms = 0.0;
    double f32_to_u8_ms = 0.0;
    double write_ms = 0.0;
    double total_ms = 0.0;
};

static inline int clamp_i(int x, int lo, int hi) {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

static inline float clamp01(float x) {
    if (x < 0.0f) {
        return 0.0f;
    }
    if (x > 1.0f) {
        return 1.0f;
    }
    return x;
}

static double ms_since(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static std::string next_token(std::istream& in) {
    std::string token;
    while (in >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string ignored;
            std::getline(in, ignored);
            continue;
        }
        return token;
    }
    throw std::runtime_error("Unexpected end of file while parsing PPM header");
}

static Image read_ppm(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open input frame: " + path.string());
    }

    const std::string magic = next_token(in);
    if (magic != "P6") {
        throw std::runtime_error("Only P6 PPM is supported: " + path.string());
    }

    const int width = std::stoi(next_token(in));
    const int height = std::stoi(next_token(in));
    const int maxval = std::stoi(next_token(in));
    if (width <= 0 || height <= 0 || maxval != 255) {
        throw std::runtime_error("Invalid PPM header: " + path.string());
    }

    in.get();  // consume one trailing whitespace byte before pixel data

    Image img;
    img.width = width;
    img.height = height;
    img.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3ULL);

    in.read(reinterpret_cast<char*>(img.pixels.data()), static_cast<std::streamsize>(img.pixels.size()));
    if (!in) {
        throw std::runtime_error("Failed to read pixel payload: " + path.string());
    }
    return img;
}

static void write_ppm(const fs::path& path, int width, int height, const std::vector<unsigned char>& pixels) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output frame: " + path.string());
    }
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    if (!out) {
        throw std::runtime_error("Failed to write output frame: " + path.string());
    }
}

static std::vector<fs::path> list_ppm_frames(const std::string& folder) {
    if (!fs::exists(folder)) {
        throw std::runtime_error("Input directory does not exist: " + folder);
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto ext = entry.path().extension().string();
        if (ext == ".ppm" || ext == ".PPM") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("No .ppm frames found in: " + folder);
    }
    return files;
}

static int default_target_height(int in_h) {
    if (in_h == 360) {
        return 720;
    }
    if (in_h == 480) {
        return 720;
    }
    if (in_h == 720) {
        return 1080;
    }
    throw std::runtime_error("Input height must be exactly 360, 480, or 720");
}

static void validate_target(int in_h, int target_h) {
    if (in_h == 360 && (target_h == 480 || target_h == 720 || target_h == 1080)) {
        return;
    }
    if (in_h == 480 && target_h == 720) {
        return;
    }
    if (in_h == 720 && target_h == 1080) {
        return;
    }
    std::ostringstream oss;
    oss << "Invalid target height " << target_h << " for input height " << in_h;
    throw std::runtime_error(oss.str());
}

static void print_usage(const char* exe_name) {
    std::cerr
        << "Usage:\n"
        << "  " << exe_name << " --input_dir <frames_ppm> --output_dir <frames_ppm>\n"
        << "      [--backend cuda|cpu] [--target_height <int>]\n"
        << "      [--w_prev <float>] [--w_curr <float>] [--w_next <float>]\n"
        << "      [--sharpen <float>] [--metrics <path.json>]\n";
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (key == "--input_dir") {
            args.input_dir = need_value("--input_dir");
        } else if (key == "--output_dir") {
            args.output_dir = need_value("--output_dir");
        } else if (key == "--backend") {
            args.backend = need_value("--backend");
        } else if (key == "--target_height") {
            args.target_height = std::stoi(need_value("--target_height"));
        } else if (key == "--w_prev") {
            args.w_prev = std::stof(need_value("--w_prev"));
        } else if (key == "--w_curr") {
            args.w_curr = std::stof(need_value("--w_curr"));
        } else if (key == "--w_next") {
            args.w_next = std::stof(need_value("--w_next"));
        } else if (key == "--sharpen") {
            args.sharpen = std::stof(need_value("--sharpen"));
        } else if (key == "--metrics") {
            args.metrics_path = need_value("--metrics");
        } else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }

    if (args.input_dir.empty() || args.output_dir.empty()) {
        throw std::runtime_error("--input_dir and --output_dir are required");
    }
    if (args.backend != "cuda" && args.backend != "cpu") {
        throw std::runtime_error("--backend must be either 'cuda' or 'cpu'");
    }

    const float w_sum = args.w_prev + args.w_curr + args.w_next;
    if (std::fabs(w_sum) < 1e-8f) {
        throw std::runtime_error("Fusion weights sum to zero");
    }
    args.w_prev /= w_sum;
    args.w_curr /= w_sum;
    args.w_next /= w_sum;
    return args;
}

static void u8_to_f32_cpu(const std::vector<unsigned char>& in, std::vector<float>& out) {
    const size_t n = in.size();
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(in[i]) * (1.0f / 255.0f);
    }
}

static void f32_to_u8_cpu(const std::vector<float>& in, std::vector<unsigned char>& out) {
    const size_t n = in.size();
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<unsigned char>(clamp01(in[i]) * 255.0f + 0.5f);
    }
}

static void temporal_fusion_cpu(
    const std::vector<float>& prev,
    const std::vector<float>& curr,
    const std::vector<float>& next,
    std::vector<float>& fused,
    float w_prev,
    float w_curr,
    float w_next
) {
    const size_t n = curr.size();
    fused.resize(n);
    for (size_t i = 0; i < n; ++i) {
        fused[i] = w_prev * prev[i] + w_curr * curr[i] + w_next * next[i];
    }
}

static void resize_bilinear_cpu(
    const std::vector<float>& in,
    int in_w,
    int in_h,
    std::vector<float>& out,
    int out_w,
    int out_h
) {
    out.resize(static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * 3ULL);
    for (int y = 0; y < out_h; ++y) {
        const float src_y = ((static_cast<float>(y) + 0.5f) * static_cast<float>(in_h) / static_cast<float>(out_h)) - 0.5f;
        int y0 = static_cast<int>(std::floor(src_y));
        const float ay = src_y - std::floor(src_y);
        y0 = clamp_i(y0, 0, in_h - 1);
        const int y1 = std::min(y0 + 1, in_h - 1);

        for (int x = 0; x < out_w; ++x) {
            const float src_x = ((static_cast<float>(x) + 0.5f) * static_cast<float>(in_w) / static_cast<float>(out_w)) - 0.5f;
            int x0 = static_cast<int>(std::floor(src_x));
            const float ax = src_x - std::floor(src_x);
            x0 = clamp_i(x0, 0, in_w - 1);
            const int x1 = std::min(x0 + 1, in_w - 1);

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
    }
}

static void sharpen_cpu(const std::vector<float>& in, int w, int h, std::vector<float>& out, float amount) {
    out.resize(in.size());
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
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
                const float v = in[center + c];
                const float blur = (4.0f * v + in[left + c] + in[right + c] + in[up + c] + in[down + c]) * (1.0f / 8.0f);
                out[center + c] = clamp01(v + amount * (v - blur));
            }
        }
    }
}

static void write_metrics_json(
    const std::string& path,
    const Args& args,
    int in_w,
    int in_h,
    int out_w,
    int out_h,
    const Metrics& m
) {
    if (path.empty()) {
        return;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open metrics file: " + path);
    }

    const double mean_total = (m.frames > 0) ? (m.total_ms / static_cast<double>(m.frames)) : 0.0;
    const double fps = (mean_total > 0.0) ? (1000.0 / mean_total) : 0.0;

    out << std::fixed << std::setprecision(4);
    out << "{\n";
    out << "  \"backend\": \"" << args.backend << "\",\n";
    out << "  \"frames\": " << m.frames << ",\n";
    out << "  \"input_width\": " << in_w << ",\n";
    out << "  \"input_height\": " << in_h << ",\n";
    out << "  \"output_width\": " << out_w << ",\n";
    out << "  \"output_height\": " << out_h << ",\n";
    out << "  \"weights\": {\n";
    out << "    \"prev\": " << args.w_prev << ",\n";
    out << "    \"curr\": " << args.w_curr << ",\n";
    out << "    \"next\": " << args.w_next << "\n";
    out << "  },\n";
    out << "  \"timing_mean_ms\": {\n";
    out << "    \"load\": " << (m.frames ? m.load_ms / m.frames : 0.0) << ",\n";
    out << "    \"h2d\": " << (m.frames ? m.h2d_ms / m.frames : 0.0) << ",\n";
    out << "    \"u8_to_f32\": " << (m.frames ? m.u8_to_f32_ms / m.frames : 0.0) << ",\n";
    out << "    \"temporal\": " << (m.frames ? m.temporal_ms / m.frames : 0.0) << ",\n";
    out << "    \"resize\": " << (m.frames ? m.resize_ms / m.frames : 0.0) << ",\n";
    out << "    \"sharpen\": " << (m.frames ? m.sharpen_ms / m.frames : 0.0) << ",\n";
    out << "    \"f32_to_u8\": " << (m.frames ? m.f32_to_u8_ms / m.frames : 0.0) << ",\n";
    out << "    \"d2h\": " << (m.frames ? m.d2h_ms / m.frames : 0.0) << ",\n";
    out << "    \"write\": " << (m.frames ? m.write_ms / m.frames : 0.0) << ",\n";
    out << "    \"total\": " << mean_total << "\n";
    out << "  },\n";
    out << "  \"fps_est\": " << fps << "\n";
    out << "}\n";
}

static void process_cpu(
    const Args& args,
    const std::vector<fs::path>& files,
    int in_w,
    int in_h,
    int out_w,
    int out_h,
    Metrics& m
) {
    std::vector<float> prev_f;
    std::vector<float> curr_f;
    std::vector<float> next_f;
    std::vector<float> fused_f;
    std::vector<float> upscaled_f;
    std::vector<float> sharp_f;
    std::vector<unsigned char> out_u8;

    const int n = static_cast<int>(files.size());
    for (int i = 0; i < n; ++i) {
        const auto total_start = Clock::now();
        const int p = (i == 0) ? 0 : (i - 1);
        const int q = i;
        const int r = (i + 1 < n) ? (i + 1) : (n - 1);

        auto t = Clock::now();
        const Image prev = read_ppm(files[p]);
        const Image curr = read_ppm(files[q]);
        const Image next = read_ppm(files[r]);
        m.load_ms += ms_since(t);

        if (prev.width != in_w || prev.height != in_h || curr.width != in_w || curr.height != in_h || next.width != in_w || next.height != in_h) {
            throw std::runtime_error("All frames must have identical dimensions");
        }

        t = Clock::now();
        u8_to_f32_cpu(prev.pixels, prev_f);
        u8_to_f32_cpu(curr.pixels, curr_f);
        u8_to_f32_cpu(next.pixels, next_f);
        m.u8_to_f32_ms += ms_since(t);

        t = Clock::now();
        temporal_fusion_cpu(prev_f, curr_f, next_f, fused_f, args.w_prev, args.w_curr, args.w_next);
        m.temporal_ms += ms_since(t);

        t = Clock::now();
        resize_bilinear_cpu(fused_f, in_w, in_h, upscaled_f, out_w, out_h);
        m.resize_ms += ms_since(t);

        t = Clock::now();
        sharpen_cpu(upscaled_f, out_w, out_h, sharp_f, args.sharpen);
        m.sharpen_ms += ms_since(t);

        t = Clock::now();
        f32_to_u8_cpu(sharp_f, out_u8);
        m.f32_to_u8_ms += ms_since(t);

        t = Clock::now();
        std::ostringstream frame_name;
        frame_name << "frame_" << std::setw(6) << std::setfill('0') << (i + 1) << ".ppm";
        write_ppm(fs::path(args.output_dir) / frame_name.str(), out_w, out_h, out_u8);
        m.write_ms += ms_since(t);

        m.total_ms += ms_since(total_start);
        m.frames += 1;
    }
}

static float time_cuda_stage(
    cudaEvent_t begin_evt,
    cudaEvent_t end_evt,
    const std::function<void()>& stage
) {
    check_cuda(cudaEventRecord(begin_evt), "cudaEventRecord(begin)");
    stage();
    check_cuda(cudaEventRecord(end_evt), "cudaEventRecord(end)");
    check_cuda(cudaEventSynchronize(end_evt), "cudaEventSynchronize(end)");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, begin_evt, end_evt), "cudaEventElapsedTime");
    return ms;
}

static void process_cuda(
    const Args& args,
    const std::vector<fs::path>& files,
    int in_w,
    int in_h,
    int out_w,
    int out_h,
    Metrics& m
) {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA device found. Use --backend cpu to run baseline path.");
    }

    const size_t in_bytes = static_cast<size_t>(in_w) * static_cast<size_t>(in_h) * 3ULL;
    const size_t in_float_bytes = in_bytes * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * 3ULL;
    const size_t out_float_bytes = out_bytes * sizeof(float);
    const int in_numel = static_cast<int>(in_bytes);
    const int out_numel = static_cast<int>(out_bytes);

    unsigned char* d_prev_u8 = nullptr;
    unsigned char* d_curr_u8 = nullptr;
    unsigned char* d_next_u8 = nullptr;
    unsigned char* d_out_u8 = nullptr;
    float* d_prev_f = nullptr;
    float* d_curr_f = nullptr;
    float* d_next_f = nullptr;
    float* d_fused = nullptr;
    float* d_upscaled = nullptr;
    float* d_sharp = nullptr;

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_prev_u8), in_bytes), "cudaMalloc d_prev_u8");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_curr_u8), in_bytes), "cudaMalloc d_curr_u8");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_next_u8), in_bytes), "cudaMalloc d_next_u8");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out_u8), out_bytes), "cudaMalloc d_out_u8");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_prev_f), in_float_bytes), "cudaMalloc d_prev_f");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_curr_f), in_float_bytes), "cudaMalloc d_curr_f");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_next_f), in_float_bytes), "cudaMalloc d_next_f");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_fused), in_float_bytes), "cudaMalloc d_fused");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_upscaled), out_float_bytes), "cudaMalloc d_upscaled");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_sharp), out_float_bytes), "cudaMalloc d_sharp");

    cudaEvent_t e0 = nullptr;
    cudaEvent_t e1 = nullptr;
    check_cuda(cudaEventCreate(&e0), "cudaEventCreate(e0)");
    check_cuda(cudaEventCreate(&e1), "cudaEventCreate(e1)");

    std::vector<unsigned char> out_host(out_bytes);

    const int n = static_cast<int>(files.size());
    for (int i = 0; i < n; ++i) {
        const auto total_start = Clock::now();
        const int p = (i == 0) ? 0 : (i - 1);
        const int q = i;
        const int r = (i + 1 < n) ? (i + 1) : (n - 1);

        auto t = Clock::now();
        const Image prev = read_ppm(files[p]);
        const Image curr = read_ppm(files[q]);
        const Image next = read_ppm(files[r]);
        m.load_ms += ms_since(t);

        if (prev.width != in_w || prev.height != in_h || curr.width != in_w || curr.height != in_h || next.width != in_w || next.height != in_h) {
            throw std::runtime_error("All frames must have identical dimensions");
        }

        t = Clock::now();
        check_cuda(cudaMemcpy(d_prev_u8, prev.pixels.data(), in_bytes, cudaMemcpyHostToDevice), "cudaMemcpy prev H2D");
        check_cuda(cudaMemcpy(d_curr_u8, curr.pixels.data(), in_bytes, cudaMemcpyHostToDevice), "cudaMemcpy curr H2D");
        check_cuda(cudaMemcpy(d_next_u8, next.pixels.data(), in_bytes, cudaMemcpyHostToDevice), "cudaMemcpy next H2D");
        m.h2d_ms += ms_since(t);

        m.u8_to_f32_ms += time_cuda_stage(e0, e1, [&]() {
            launch_u8_to_f32(d_prev_u8, d_prev_f, in_numel);
            launch_u8_to_f32(d_curr_u8, d_curr_f, in_numel);
            launch_u8_to_f32(d_next_u8, d_next_f, in_numel);
        });

        m.temporal_ms += time_cuda_stage(e0, e1, [&]() {
            launch_temporal_fusion(d_prev_f, d_curr_f, d_next_f, d_fused, in_numel, args.w_prev, args.w_curr, args.w_next);
        });

        m.resize_ms += time_cuda_stage(e0, e1, [&]() {
            launch_resize_bilinear_rgb(d_fused, in_w, in_h, d_upscaled, out_w, out_h);
        });

        m.sharpen_ms += time_cuda_stage(e0, e1, [&]() {
            launch_sharpen_rgb(d_upscaled, d_sharp, out_w, out_h, args.sharpen);
        });

        m.f32_to_u8_ms += time_cuda_stage(e0, e1, [&]() {
            launch_f32_to_u8(d_sharp, d_out_u8, out_numel);
        });

        t = Clock::now();
        check_cuda(cudaMemcpy(out_host.data(), d_out_u8, out_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy out D2H");
        m.d2h_ms += ms_since(t);

        t = Clock::now();
        std::ostringstream frame_name;
        frame_name << "frame_" << std::setw(6) << std::setfill('0') << (i + 1) << ".ppm";
        write_ppm(fs::path(args.output_dir) / frame_name.str(), out_w, out_h, out_host);
        m.write_ms += ms_since(t);

        m.total_ms += ms_since(total_start);
        m.frames += 1;
    }

    check_cuda(cudaEventDestroy(e0), "cudaEventDestroy(e0)");
    check_cuda(cudaEventDestroy(e1), "cudaEventDestroy(e1)");
    check_cuda(cudaFree(d_prev_u8), "cudaFree d_prev_u8");
    check_cuda(cudaFree(d_curr_u8), "cudaFree d_curr_u8");
    check_cuda(cudaFree(d_next_u8), "cudaFree d_next_u8");
    check_cuda(cudaFree(d_out_u8), "cudaFree d_out_u8");
    check_cuda(cudaFree(d_prev_f), "cudaFree d_prev_f");
    check_cuda(cudaFree(d_curr_f), "cudaFree d_curr_f");
    check_cuda(cudaFree(d_next_f), "cudaFree d_next_f");
    check_cuda(cudaFree(d_fused), "cudaFree d_fused");
    check_cuda(cudaFree(d_upscaled), "cudaFree d_upscaled");
    check_cuda(cudaFree(d_sharp), "cudaFree d_sharp");
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::vector<fs::path> files = list_ppm_frames(args.input_dir);

        const Image first = read_ppm(files[0]);
        const int in_w = first.width;
        const int in_h = first.height;

        if (in_h != 360 && in_h != 480 && in_h != 720) {
            throw std::runtime_error("Input height must be exactly 360, 480, or 720");
        }

        int target_h = args.target_height;
        if (target_h <= 0) {
            target_h = default_target_height(in_h);
        }
        validate_target(in_h, target_h);

        const int out_w = static_cast<int>(std::llround(static_cast<double>(in_w) * static_cast<double>(target_h) / static_cast<double>(in_h)));
        const int out_h = target_h;

        fs::create_directories(args.output_dir);

        std::cout << "Input frames:   " << files.size() << "\n";
        std::cout << "Input size:     " << in_w << "x" << in_h << "\n";
        std::cout << "Output size:    " << out_w << "x" << out_h << "\n";
        std::cout << "Backend:        " << args.backend << "\n";
        std::cout << "Weights:        prev=" << args.w_prev << ", curr=" << args.w_curr << ", next=" << args.w_next << "\n";
        std::cout << "Sharpen amount: " << args.sharpen << "\n";

        Metrics metrics;
        if (args.backend == "cuda") {
            process_cuda(args, files, in_w, in_h, out_w, out_h, metrics);
        } else {
            process_cpu(args, files, in_w, in_h, out_w, out_h, metrics);
        }

        write_metrics_json(args.metrics_path, args, in_w, in_h, out_w, out_h, metrics);

        const double mean_total = metrics.frames ? metrics.total_ms / metrics.frames : 0.0;
        const double fps_est = mean_total > 0.0 ? (1000.0 / mean_total) : 0.0;
        std::cout << "Processed frames: " << metrics.frames << "\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Mean total/frame: " << mean_total << " ms\n";
        std::cout << "Estimated FPS:    " << fps_est << "\n";
        if (!args.metrics_path.empty()) {
            std::cout << "Metrics:          " << args.metrics_path << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
}
