// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/kernels/transpose_kernel.h"

#ifdef _OPENMP
#include <omp.h>
#else
// 定义空宏防止编译错误
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

namespace {

template <typename T>
inline T xabs(const T x) {
  return x < static_cast<T>(0.0) ? -x : x;
}

template <typename T, typename ScaleT>
void per_channel_scale(
    ScaleT* scale, const T* input, size_t m, size_t n, float bound) {
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    float max = static_cast<float>(input[i]);
    for (size_t j = 0; j < m; ++j) {
      max = static_cast<float>(xabs(input[j * n + i])) > max
                ? static_cast<float>(xabs(input[j * n + i]))
                : max;
    }
    scale[i] = static_cast<ScaleT>(max / bound);
  }
}

template <typename T, typename ScaleT>
void group_wise_scale(ScaleT* scale,
                      const T* input,
                      size_t m,
                      size_t n,
                      float bound,
                      size_t group_size) {
  const size_t num_blocks = (m + group_size - 1) / group_size;

#pragma omp parallel for collapse(2) schedule(static)
  for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
    for (size_t i = 0; i < n; ++i) {
      const size_t j_start = block_idx * group_size;
      const size_t j_end = std::min(j_start + group_size, m);
      float max_val = 0.0f;
#pragma omp simd reduction(max : max_val) aligned(input : 64)
      for (size_t j = j_start; j < j_end; ++j) {
        const float abs_val = xabs(input[j * n + i]);
        if (abs_val > max_val) max_val = abs_val;
      }

      scale[block_idx * n + i] = static_cast<ScaleT>(max_val / bound);
    }
  }
}

template <typename T, int quant_bit = 8, typename ScaleT>
void per_channel_quant(int8_t* output,
                       const T* input,
                       const ScaleT* scale,
                       size_t num_rows,
                       size_t num_cols) {
  size_t bytes_per_out_col = num_cols * quant_bit / 8;
#pragma omp parallel for
  for (size_t ii = 0; ii < num_rows; ++ii) {
    int8_t* current_quantized_weight_row = output + ii * bytes_per_out_col;
    const T* current_weight_row = input + ii * num_cols;
    for (size_t jj = 0; jj < bytes_per_out_col; ++jj) {
      if (quant_bit == 8) {
        const float col_scale = static_cast<float>(scale[jj]);
        const float weight_elt = static_cast<float>(current_weight_row[jj]);
        const float scaled_weight = round(weight_elt / col_scale);
        const int8_t clipped_weight = static_cast<int8_t>(
            std::max(-127.f, std::min(127.f, scaled_weight)));
        current_quantized_weight_row[jj] = clipped_weight;
      } else if (quant_bit == 4) {
        // We will pack two int4 elements per iteration of the inner loop.
        int8_t packed_int4s = 0;
        for (int packed_idx = 0; packed_idx < 2; ++packed_idx) {
          const size_t input_idx = 2 * jj + packed_idx;
          if (input_idx < num_cols) {
            const float col_scale = static_cast<float>(scale[input_idx]);
            const float weight_elt =
                static_cast<float>(current_weight_row[input_idx]);
            const float scaled_weight = round(weight_elt / col_scale);
            int int_weight = static_cast<int>(scaled_weight);
            const int8_t clipped_weight =
                std::max(-7, std::min(7, int_weight)) + 8;

            // Kill the sign extension bits (hence 0x0F mask) then shift to
            // upper bits if packing the second int4 and or the bits into the
            // final result.
            packed_int4s |= ((clipped_weight & 0x0F) << (4 * packed_idx));
          }
        }
        current_quantized_weight_row[jj] = packed_int4s;
      } else {
        common::errors::Unimplemented("Unsupported quantization bits: %d",
                                      quant_bit);
      }
    }
  }
}

template <typename T, int quant_bit = 8, typename ScaleT>
void group_wise_quant(int8_t* output,
                      const T* input,
                      const ScaleT* scale,
                      size_t num_rows,
                      size_t num_cols,
                      const int group_size) {
  const size_t bytes_per_out_col = num_cols * quant_bit / 8;
  const size_t groups_per_row = (num_rows + group_size - 1) / group_size;

  // 预处理缩放因子，避免重复计算
  std::vector<float> inv_scale(num_cols * groups_per_row);
#pragma omp parallel for collapse(2)
  for (size_t g = 0; g < groups_per_row; ++g) {
    for (size_t col = 0; col < num_cols; ++col) {
      const size_t idx = g * num_cols + col;
      inv_scale[idx] = 1.0f / static_cast<float>(scale[idx]);
    }
  }

#pragma omp parallel for schedule(dynamic, 4)
  for (size_t ii = 0; ii < num_rows; ++ii) {
    const size_t group_idx = ii / group_size;
    const float* group_inv_scale = inv_scale.data() + group_idx * num_cols;
    const T* current_weight_row = input + ii * num_cols;
    int8_t* quant_row = output + ii * bytes_per_out_col;

    if constexpr (quant_bit == 8) {
      // 8-bit 量化向量化处理
      constexpr int simd_width = 32;  // AVX2: 8 floats, AVX512: 16 floats
      const size_t vec_cols = num_cols / simd_width * simd_width;

// 主向量循环
#pragma omp simd
      for (size_t jj = 0; jj < vec_cols; jj += simd_width) {
        alignas(64) float scaled_weights[simd_width];

        // 向量化加载和计算
        for (int k = 0; k < simd_width; ++k) {
          const float weight_val =
              static_cast<float>(current_weight_row[jj + k]);
          scaled_weights[k] = std::round(weight_val * group_inv_scale[jj + k]);
        }

        // 向量化截断和存储
        for (int k = 0; k < simd_width; ++k) {
          scaled_weights[k] =
              std::max(-127.0f, std::min(127.0f, scaled_weights[k]));
          quant_row[jj + k] = static_cast<int8_t>(scaled_weights[k]);
        }
      }

      // 处理尾部元素
      for (size_t jj = vec_cols; jj < num_cols; ++jj) {
        const float weight_val = static_cast<float>(current_weight_row[jj]);
        float scaled_weight = std::round(weight_val * group_inv_scale[jj]);
        scaled_weight = std::max(-127.0f, std::min(127.0f, scaled_weight));
        quant_row[jj] = static_cast<int8_t>(scaled_weight);
      }
    } else if constexpr (quant_bit == 4) {
      // 4-bit 量化优化处理
      const size_t packed_cols = num_cols / 2;
#pragma omp simd
      for (size_t jj = 0; jj < packed_cols; ++jj) {
        const size_t idx0 = 2 * jj;
        const size_t idx1 = 2 * jj + 1;

        // 同时处理两个元素
        const float weight0 = static_cast<float>(current_weight_row[idx0]);
        const float weight1 = static_cast<float>(current_weight_row[idx1]);

        float scaled0 = std::round(weight0 * group_inv_scale[idx0]);
        float scaled1 = std::round(weight1 * group_inv_scale[idx1]);

        int int0 = static_cast<int>(std::max(-7.0f, std::min(7.0f, scaled0)));
        int int1 = static_cast<int>(std::max(-7.0f, std::min(7.0f, scaled1)));

        // 调整范围并打包
        int0 = (int0 + 8) & 0x0F;
        int1 = (int1 + 8) & 0x0F;
        quant_row[jj] = static_cast<int8_t>(int0 | (int1 << 4));
      }

      // 处理奇数列情况
      if (num_cols % 2 != 0) {
        const size_t last_idx = num_cols - 1;
        const size_t jj = packed_cols;
        const float weight_val =
            static_cast<float>(current_weight_row[last_idx]);
        float scaled_weight =
            std::round(weight_val * group_inv_scale[last_idx]);
        int int_weight =
            static_cast<int>(std::max(-7.0f, std::min(7.0f, scaled_weight)));
        int_weight = (int_weight + 8) & 0x0F;
        quant_row[jj] = static_cast<int8_t>(int_weight);
      }
    } else {
      static_assert(quant_bit == 4 || quant_bit == 8,
                    "Unsupported quantization bits");
    }
  }
}

}  // namespace

namespace custom_kernel {

template <typename DeviceContext,
          typename T,
          typename D,
          int bits,
          typename ScaleT = T>
void quant_compute(const DeviceContext& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out,
                   phi::DenseTensor* scale,
                   const std::string& algo,
                   const int32_t arch,
                   const int32_t group_size) {
  const auto x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2,
      common::errors::InvalidArgument(
          "the x tensor of quant op must be 2D, but got[%d]", x_dims.size()));
  size_t m = x_dims[0];
  size_t n = x_dims[1];
  int64_t num = x.numel();
  phi::DDim dims = {num};
  const T* x_data = x.data<T>();
  ScaleT* scale_data = scale->data<ScaleT>();

  phi::DenseTensorMeta out_meta = out->meta();
  phi::DenseTensor x_int;
  x_int.set_meta(out_meta);
  x_int.Resize({static_cast<int64_t>(m), static_cast<int64_t>(n)});
  dev_ctx.template Alloc<D>(&x_int);
  D* x_int_data = x_int.data<D>();

  phi::DenseTensor x_int_tmp;
  x_int_tmp.set_meta(out_meta);
  x_int_tmp.Resize({static_cast<int64_t>(m), static_cast<int64_t>(n / 2)});
  dev_ctx.template Alloc<D>(&x_int_tmp);
  D* x_int_tmp_data = x_int_tmp.data<D>();

  if (group_size == -1) {
    per_channel_scale(scale_data, x_data, m, n, bits == 8 ? 127.0f : 7.0f);
    per_channel_quant<T, bits>(x_int_data, x_data, scale_data, m, n);
  } else {
    group_wise_scale(scale_data,
                     x_data,
                     m,
                     n,
                     bits == 8 ? 127.0f : 7.0f,
                     static_cast<size_t>(group_size));
    group_wise_quant<T, bits>(x_int_data, x_data, scale_data, m, n, group_size);
  }

  if (bits == 8) {
    std::vector<int> axis = {1, 0};
    phi::funcs::Transpose<DeviceContext, int8_t, 2> trans;
    trans(dev_ctx, x_int, out, axis);
  } else {
    for (int i = 0; i < out->numel(); ++i) {
      x_int_tmp_data[i] = x_int_data[i];
    }
    std::vector<int> axis = {1, 0};
    phi::funcs::Transpose<DeviceContext, int8_t, 2> trans;
    trans(dev_ctx, x_int_tmp, out, axis);
  }
}

template <typename T, typename Context>
void WeightQuantizeKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const std::string& algo,
                          const int32_t arch,
                          const int32_t group_size,
                          phi::DenseTensor* out,
                          phi::DenseTensor* scale) {
  PADDLE_GCU_KERNEL_TRACE("weight_quantize");
  phi::DenseTensor x_cpu;
  phi::DenseTensor out_cpu;
  phi::DenseTensor scale_cpu;
  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.SetAllocator(&(dev_ctx.GetHostAllocator()));
  dev_ctx_cpu.SetHostAllocator(&(dev_ctx.GetHostAllocator()));
  TensorCopy(dev_ctx, x, true, &x_cpu, phi::CPUPlace());
  dev_ctx.Wait();

  phi::DenseTensorMeta out_meta = out->meta();
  out_cpu.set_meta(out_meta);
  dev_ctx_cpu.template Alloc<int8_t>(&out_cpu);
  phi::DenseTensorMeta scale_meta = scale->meta();
  scale_cpu.set_meta(scale_meta);
  if (algo == "weight_only_int8") {
    dev_ctx_cpu.template Alloc<T>(&scale_cpu);
    quant_compute<phi::CPUContext, T, int8_t, 8>(
        dev_ctx_cpu, x_cpu, &out_cpu, &scale_cpu, algo, arch, group_size);
  } else if (algo == "llm.int8") {
    dev_ctx_cpu.template Alloc<float>(&scale_cpu);
    quant_compute<phi::CPUContext, T, int8_t, 8, float>(
        dev_ctx_cpu, x_cpu, &out_cpu, &scale_cpu, algo, arch, group_size);
  } else if (algo == "weight_only_int4") {
    dev_ctx_cpu.template Alloc<T>(&scale_cpu);
    quant_compute<phi::CPUContext, T, int8_t, 4>(
        dev_ctx_cpu, x_cpu, &out_cpu, &scale_cpu, algo, arch, group_size);
  } else {
    common::errors::Unimplemented(
        "The algo must be in ['weight_only_int8', 'weight_only_int4', "
        "'llm.int8'], but got[%s]",
        algo);
  }
  dev_ctx_cpu.Wait();
  TensorCopy(dev_ctx, out_cpu, true, out);
  TensorCopy(dev_ctx, scale_cpu, true, scale);
  dev_ctx.Wait();
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(weight_quantize,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::WeightQuantizeKernel,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}
