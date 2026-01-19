// PaddleCustomDevice/backends/metax_gpu/cinn/compiler/compiler.cc

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <ctime>
#include <atomic>

// Host 端头文件，仅供 compiler.cc 使用
#include "paddle/phi/backends/device_ext.h"

namespace paddle {
namespace custom_device {
namespace metax {

// ============================================================
// 1. Runtime Source (JIT 源码头文件 - Device 端代码)
// ============================================================
static const char* kMacaRuntimeSource = R"MACA_SOURCE(
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>

extern "C" {

#define WARP_SIZE 64

#if defined(__MACACC_RTC__) || defined(__HIPCC_RTC__) || defined(__CUDACC_RTC__)
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
#endif

// ===============================================================
// Float64 (Double) Math Functions
// ===============================================================
#define FN_FP64(func) cinn_custom_device_##func##_fp64

__device__ inline double FN_FP64(sin)(double x) { return sin(x); }
__device__ inline double FN_FP64(cos)(double x) { return cos(x); }
__device__ inline double FN_FP64(tan)(double x) { return tan(x); }
__device__ inline double FN_FP64(exp)(double x) { return exp(x); }
__device__ inline double FN_FP64(log)(double x) { return log(x); }
__device__ inline double FN_FP64(log2)(double x) { return log2(x); }
__device__ inline double FN_FP64(log10)(double x) { return log10(x); }
__device__ inline double FN_FP64(sqrt)(double x) { return sqrt(x); }
__device__ inline double FN_FP64(rsqrt)(double x) { return rsqrt(x); }
__device__ inline double FN_FP64(abs)(double x) { return fabs(x); }
__device__ inline double FN_FP64(floor)(double x) { return floor(x); }
__device__ inline double FN_FP64(ceil)(double x) { return ceil(x); }
__device__ inline double FN_FP64(round)(double x) { return round(x); }
__device__ inline double FN_FP64(trunc)(double x) { return trunc(x); }
__device__ inline double FN_FP64(pow)(double a, double b) { return pow(a, b); }
__device__ inline double FN_FP64(mod)(double a, double b) { return fmod(a, b); }
__device__ inline bool FN_FP64(isnan)(double x) { return isnan(x); }
__device__ inline bool FN_FP64(isinf)(double x) { return isinf(x); }
__device__ inline bool FN_FP64(isfinite)(double x) { return isfinite(x); }

// ===============================================================
// Float32 Math Functions
// ===============================================================
#define FN_FP32(func) cinn_custom_device_##func##_fp32

__device__ inline float FN_FP32(sin)(float x) { return sinf(x); }
__device__ inline float FN_FP32(cos)(float x) { return cosf(x); }
__device__ inline float FN_FP32(tan)(float x) { return tanf(x); }
__device__ inline float FN_FP32(exp)(float x) { return expf(x); }
__device__ inline float FN_FP32(log)(float x) { return logf(x); }
__device__ inline float FN_FP32(sqrt)(float x) { return sqrtf(x); }
__device__ inline float FN_FP32(rsqrt)(float x) { return rsqrtf(x); }
__device__ inline float FN_FP32(pow)(float a, float b) { return powf(a, b); }
__device__ inline float FN_FP32(floor)(float x) { return floorf(x); }
__device__ inline float FN_FP32(ceil)(float x) { return ceilf(x); }
__device__ inline float FN_FP32(round)(float x) { return roundf(x); }
__device__ inline float FN_FP32(trunc)(float x) { return truncf(x); }
__device__ inline float FN_FP32(abs)(float x) { return fabsf(x); }

// ===============================================================
// Bool / Int logic
// ===============================================================
#define FN_BOOL(func) cinn_custom_device_##func##_bool
__device__ inline bool FN_BOOL(bitwise_and)(bool a, bool b) { return a & b; }
__device__ inline bool FN_BOOL(bitwise_or)(bool a, bool b) { return a | b; }
__device__ inline bool FN_BOOL(bitwise_not)(bool a) { return !a; }
__device__ inline bool FN_BOOL(bitwise_xor)(bool a, bool b) { return a ^ b; }

// ===============================================================
// Int32 Functions
// ===============================================================
#define FN_INT32(func) cinn_custom_device_##func##_int32
__device__ inline int FN_INT32(bitwise_not)(int a) { return ~a; }
__device__ inline int FN_INT32(clz)(int a) { return __clz(a); }
__device__ inline int FN_INT32(popc)(int a) { return __popc(a); }
__device__ inline int FN_INT32(mod)(int a, int b) { 
  int res = a % b;
  if ((res != 0) && ((b ^ res) < 0)) res += b;
  return res;
}

// ===============================================================
// Float16 (Half) Functions
// ===============================================================
#define FN_FP16(func) cinn_custom_device_##func##_fp16

__device__ inline __half FN_FP16(ceil)(__half x) { return hceil(x); }
__device__ inline __half FN_FP16(floor)(__half x) { return hfloor(x); }
__device__ inline __half FN_FP16(sin)(__half x) { return hsin(x); }
__device__ inline __half FN_FP16(cos)(__half x) { return hcos(x); }
__device__ inline __half FN_FP16(exp)(__half x) { return hexp(x); }
__device__ inline __half FN_FP16(log)(__half x) { return hlog(x); }
__device__ inline __half FN_FP16(log2)(__half x) { return hlog2(x); }
__device__ inline __half FN_FP16(log10)(__half x) { return hlog10(x); }
__device__ inline __half FN_FP16(sqrt)(__half x) { return hsqrt(x); }
__device__ inline __half FN_FP16(rsqrt)(__half x) { return hrsqrt(x); }

// ===============================================================
// Index Operations
// ===============================================================
#define CINN_CUSTOM_DEVICE_FIND_KERNEL(buf, size, num, begin, stride) \
  do {                                                                \
    for (int i = (size - 1) * stride + begin; i >= begin; i -= stride) { \
      if (buf[i] == num) return (i - begin) / stride;                 \
    }                                                                 \
    return -1;                                                        \
  } while (0)

__device__ inline int cinn_custom_device_find_int(const int *buf, int size, int num) {
  CINN_CUSTOM_DEVICE_FIND_KERNEL(buf, size, num, 0, 1);
}
__device__ inline int cinn_custom_device_find_float(const float *buf, int size, float num) {
  CINN_CUSTOM_DEVICE_FIND_KERNEL(buf, size, num, 0, 1);
}
__device__ inline int cinn_custom_device_find_int_nd(const int *buf, int size, int num, int begin, int stride) {
  CINN_CUSTOM_DEVICE_FIND_KERNEL(buf, size, num, begin, stride);
}
__device__ inline int cinn_custom_device_find_float_nd(const float *buf, int size, float num, int begin, int stride) {
  CINN_CUSTOM_DEVICE_FIND_KERNEL(buf, size, num, begin, stride);
}

} // extern "C"
)MACA_SOURCE";


// ============================================================
// 2. 接口实现
// ============================================================

// 全局原子计数器，确保文件名唯一
static std::atomic<uint64_t> g_compile_counter{0};

const char* MetaxGetRuntimeSource(void* dev_ptr) {
    return kMacaRuntimeSource;
}

C_Status MetaxCompile(void* dev_ptr, const char* code, char* out_path, size_t len) {
    // 0. 生成随机文件名
    // 【关键修复】使用 进程ID + 原子计数器 生成唯一文件名
    // 彻底解决多线程编译时的文件名冲突问题
    uint64_t file_id = g_compile_counter.fetch_add(1);
    std::string file_prefix = "cinn_metax_" + std::to_string(getpid()) + "_" + std::to_string(file_id);
    
    // 生成临时文件路径
    std::string src_path = "/tmp/" + file_prefix + ".cu";
    std::string obj_path = "/tmp/" + file_prefix + ".co";

    // 注意：即使 CINN 传了 out_path 进来，通常也是空的或者期望我们填写的
    // 所以我们尽量使用自己生成的 obj_path，最后再拷贝回去

    // 1. 写入源码
    {
        // 使用 truncate 模式打开，虽然文件名唯一，但以防万一
        std::ofstream src_file(src_path, std::ios::trunc);
        if (!src_file.is_open()) {
            std::cerr << "[MetaX] Failed to open temp file: " << src_path << std::endl;
            return C_Status::C_FAILED;
        }
        src_file << kMacaRuntimeSource << "\n";
        src_file << code;
        src_file.close();
    }

    // 2. 准备编译器路径
    const char* maca_path_env = std::getenv("MACA_PATH");
    std::string maca_path = maca_path_env ? std::string(maca_path_env) : "/opt/maca";
    
    std::string mxcc_cmd = maca_path + "/mxgpu_llvm/bin/mxcc";
    if (access(mxcc_cmd.c_str(), X_OK) != 0) {
         mxcc_cmd = maca_path + "/bin/mxcc";
         if (access(mxcc_cmd.c_str(), X_OK) != 0) mxcc_cmd = "mxcc";
    }

    // 3. 构建编译命令
    // 注意：加了空格防止粘连
    std::string cmd = mxcc_cmd + " -O3 -std=c++17 -w --fatbin --offload-arch=native -fvisibility=default";
    cmd += " -I" + maca_path + "/include";
    cmd += " -I" + maca_path + "/tools/cu-bridge/include";
    cmd += " -o " + obj_path;
    cmd += " " + src_path;

    // 4. 执行
    std::cout << "Command: " << cmd << std::endl;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[MetaX] JIT Compilation Failed! Code: " << ret << std::endl;
        std::cerr << "Command: " << cmd << std::endl;
        return C_Status::C_FAILED;
    }

    // 5. 确保文件存在
    if (access(obj_path.c_str(), F_OK) != 0) {
        std::cerr << "[MetaX] Output file missing: " << obj_path << std::endl;
        return C_Status::C_FAILED;
    }

    // =================================================================
    // 6. 【关键修复】将生成的二进制路径回填给 CINN 框架
    // =================================================================
    if (out_path && len > 0) {
        // 使用 strncpy 安全拷贝
        std::strncpy(out_path, obj_path.c_str(), len - 1);
        out_path[len - 1] = '\0'; // 确保 null 结尾
        // 打印调试信息，确认回填成功
        std::cout << "[MetaX Success] Compiled: " << out_path << std::endl;
    } else {
        std::cerr << "[MetaX Error] Invalid out_path buffer!" << std::endl;
        return C_Status::C_FAILED;
    }

    // 7. 清理源码 (调试成功后可开启)
    std::remove(src_path.c_str());

    return C_Status::C_SUCCESS;
}

} // namespace metax
} // namespace custom_device
} // namespace paddle