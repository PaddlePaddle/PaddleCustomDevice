// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T,
          int VectorSizeInput,
          int VectorSizeOutput,
          typename ScaleT>
__global__ void per_channel_quant_gpu(const T* weight_data,
                                      int8_t* quanted_weight_data,
                                      ScaleT* scale_data,
                                      int total_k,
                                      int total_vec_k,
                                      int total_vec_n) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < total_vec_n) {
    const int* vec_weight_data_ptr = reinterpret_cast<const int*>(weight_data);
    int* vec_quanted_weight_data = reinterpret_cast<int*>(quanted_weight_data) +
                                   n * VectorSizeInput * total_vec_k;
    phi::AlignedVector<float, VectorSizeInput> abs_max;
#pragma unroll
    for (int i = 0; i < VectorSizeInput; ++i) {
      abs_max[i] = static_cast<float>(0.0f);
    }
#pragma unroll
    for (int k = 0; k < total_k; ++k) {
      int linear_index = k * total_vec_n + n;
      phi::AlignedVector<T, VectorSizeInput> weight;
      *reinterpret_cast<int*>(&weight) = vec_weight_data_ptr[linear_index];
#pragma unroll
      for (int i = 0; i < VectorSizeInput; ++i) {
        abs_max[i] = fmaxf((abs_max[i]), fabsf((weight[i])));
      }
    }
    phi::AlignedVector<ScaleT, VectorSizeInput> scale;
#pragma unroll
    for (int i = 0; i < VectorSizeInput; ++i) {
      scale[i] = static_cast<ScaleT>(abs_max[i] / static_cast<float>(127.0f));
    }
    *reinterpret_cast<float*>(scale_data + VectorSizeInput * n) =
        *reinterpret_cast<float*>(&scale);

    for (int k = 0; k < total_vec_k; ++k) {
      int8_t quanted_weight[VectorSizeInput][VectorSizeOutput];
#pragma unroll
      for (int vk_i = 0; vk_i < VectorSizeOutput; ++vk_i) {
        int linear_index = (k * VectorSizeOutput + vk_i) * total_vec_n + n;
        phi::AlignedVector<T, VectorSizeInput> weight;
        *reinterpret_cast<int*>(&weight) =
            *reinterpret_cast<const int*>(vec_weight_data_ptr + linear_index);
#pragma unroll
        for (int i = 0; i < VectorSizeInput; ++i) {
          float scaled_weight =
              (static_cast<float>(weight[i]) / static_cast<float>(abs_max[i])) *
              static_cast<float>(127.0);
          int8_t clipped_weight = static_cast<int8_t>(
              lroundf(fmaxf(-127.0f, fminf(127.0f, scaled_weight))));
          quanted_weight[i][vk_i] = clipped_weight;
        }
      }
#pragma unroll
      for (int i = 0; i < VectorSizeInput; ++i) {
        *reinterpret_cast<int*>(vec_quanted_weight_data + i * total_vec_k + k) =
            *reinterpret_cast<int*>(quanted_weight[i]);
      }
    }
  }
}

template <typename T, typename GPUContext, typename ScaleT>
void weight_quant_gpu(const GPUContext& dev_ctx,
                      const T* weight_data,
                      int8_t* quanted_weight_data,
                      ScaleT* scale_data,
                      const std::vector<int>& shape,
                      const int32_t arch,
                      const std::string& algo) {
  int total_k = shape[0];
  int total_n = shape[1];
  int numel = total_k * total_n;
  constexpr int kWarpSize = 64;
  constexpr int kBlockSize = 128;
  constexpr int kWarpNum = kBlockSize / kWarpSize;
  constexpr int VectorSizeInput = 32 / sizeof(T) / 8;
  constexpr int VectorSizeOutput = 32 / sizeof(int8_t) / 8;
  PADDLE_ENFORCE_EQ(total_n % VectorSizeInput && total_k % VectorSizeOutput,
                    0,
                    common::errors::PreconditionNotMet(
                        "Currently, weight_quant_gpu kernel only support n "
                        "with multiple of %d and k  multiple of %d, please use",
                        VectorSizeInput,
                        VectorSizeOutput));
  int vec_total_n = total_n / VectorSizeInput;
  int vec_total_k = total_k / VectorSizeOutput;
  int kGridSize = std::max((vec_total_n + kBlockSize - 1) / kBlockSize,
                           static_cast<int>(1));
  per_channel_quant_gpu<T, VectorSizeInput, VectorSizeOutput>
      <<<kGridSize, kBlockSize>>>(weight_data,
                                  quanted_weight_data,
                                  scale_data,
                                  total_k,
                                  vec_total_k,
                                  vec_total_n);
}

}  // namespace phi
