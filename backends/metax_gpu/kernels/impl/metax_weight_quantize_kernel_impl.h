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

#pragma once

#include <cstdint>

#include "paddle/common/enforce.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
template <typename DataType>
void show_2d_cpu_tensor(const DenseTensor& tensor,
                        const int64_t row_num = 3,
                        const int64_t col_num = 3) {
  const int64_t rows = tensor.dims()[0];
  const int64_t cols = tensor.dims()[1];
  printf("\nTensor shape = [%d, %d]\n", rows, cols);

  const DataType* cpu_ptr = tensor.data<DataType>();

  for (int r = 0; r < row_num; r++) {
    for (int c = 0; c < col_num; c++) {
      DataType val = *(cpu_ptr + r * cols + c);
      printf("%#x ", val);
    }
    printf("\n");
  }
  printf("\n\n");
}
template <typename DataType>
void show_2d_gpu_tensor(const CustomContext& dev_ctx,
                        const DenseTensor& tensor,
                        const int64_t row_num = 3,
                        const int64_t col_num = 3) {
  phi::CPUPlace cpu_place;

  DenseTensor cpu_tensor;
  phi::Copy(dev_ctx, tensor, cpu_place, true, &cpu_tensor);

  const int64_t rows = cpu_tensor.dims()[0];
  const int64_t cols = cpu_tensor.dims()[1];
  printf("\nTensor shape = [%d, %d]\n", rows, cols);

  const DataType* cpu_ptr = cpu_tensor.data<DataType>();

  for (int r = 0; r < row_num; r++) {
    for (int c = 0; c < col_num; c++) {
      DataType val = *(cpu_ptr + r * cols + c);
      printf("%#x ", val);
    }
    printf("\n");
  }
  printf("\n\n");
}

template <typename DataType>
void show_1d_gpu_tensor(const CustomContext& dev_ctx,
                        const DenseTensor& tensor,
                        const int64_t num = 3) {
  phi::CPUPlace cpu_place;

  DenseTensor cpu_tensor;
  phi::Copy(dev_ctx, tensor, cpu_place, true, &cpu_tensor);

  const int64_t nums = cpu_tensor.numel();
  printf("\nTensor shape = [%d]\n", nums);

  const DataType* cpu_ptr = cpu_tensor.data<DataType>();

  for (int n = 0; n < num; n++) {
    DataType val = *(cpu_ptr + n);
    printf("%#x ", val);
  }
  printf("\n\n");
}

void cpu_2d_tensor_transpose(const DenseTensor& input_data,
                             DenseTensor* transposed_data) {
  const int64_t input_data_rows = input_data.dims()[0];
  const int64_t input_data_cols = input_data.dims()[1];

  const int8_t* input_data_ptr = input_data.data<int8_t>();
  int8_t* transposed_data_ptr = transposed_data->data<int8_t>();

  for (int64_t r = 0; r < input_data_rows; r++) {
    for (int64_t c = 0; c < input_data_cols; c++) {
      *(transposed_data_ptr + r + c * input_data_rows) =
          *(input_data_ptr + r * input_data_cols + c);
    }
  }
}

void cpu_int4_quanted_weight_raw_unpack(const DenseTensor& packed_data,
                                        DenseTensor* unpacked_data) {
  const int64_t packed_data_rows = packed_data.dims()[0];
  const int64_t packed_data_cols = packed_data.dims()[1];

  const int8_t* packed_data_ptr = packed_data.data<int8_t>();
  int8_t* unpacked_data_ptr = unpacked_data->data<int8_t>();

  for (int64_t c = 0; c < packed_data_cols; c++) {
    for (int64_t r = 0; r < packed_data_rows; r++) {
      int8_t val = *(packed_data_ptr + r * packed_data_cols + c);
      int8_t low_int4 = val & 0x0f;
      int8_t hight_int4 = (val >> 4) & 0x0f;

      *(unpacked_data_ptr + (2 * r) * packed_data_cols + c) =
          low_int4 >= 8 ? low_int4 - 16 : low_int4;
      *(unpacked_data_ptr + (2 * r + 1) * packed_data_cols + c) =
          hight_int4 >= 8 ? hight_int4 - 16 : hight_int4;
    }
  }
}

void cpu_int4_quanted_weight_col_pack(const DenseTensor& unpacked_data,
                                      DenseTensor* packed_data) {
  const int64_t packed_data_rows = packed_data->dims()[0];
  const int64_t packed_data_cols = packed_data->dims()[1];

  int8_t* packed_data_ptr = packed_data->data<int8_t>();
  const int8_t* unpacked_data_ptr = unpacked_data.data<int8_t>();

  for (int64_t r = 0; r < packed_data_rows; r++) {
    for (int64_t c = 0; c < packed_data_cols; c++) {
      int8_t low_int4 = *(unpacked_data_ptr + 2 * r * packed_data_cols + 2 * c);
      int8_t hight_int4 =
          *(unpacked_data_ptr + 2 * r * packed_data_cols + 2 * c + 1);

      low_int4 = low_int4 < 0 ? low_int4 + 16 : low_int4;
      hight_int4 = hight_int4 < 0 ? hight_int4 + 16 : hight_int4;

      *(packed_data_ptr + r * packed_data_cols + c) =
          ((hight_int4 << 4) & 0xf0) | (low_int4 & 0x0f);
    }
  }
}

void cpu_int4_quantized_weight_layout_trans_impl(
    const CustomContext& dev_ctx,
    const std::vector<int64_t>& shape,
    DenseTensor* out) {
  const int64_t m = shape[0];
  const int64_t n = shape[1];

  phi::CPUPlace cpu_place;

  out->Resize({m / 2, n});

  DenseTensor out_cpu_tensor;
  phi::Copy(dev_ctx, (*out), cpu_place, true, &out_cpu_tensor);

  // raw unpack
  DenseTensor raw_unpack_tensor;
  raw_unpack_tensor.Resize({out_cpu_tensor.dims()[0] * 2, n});
  raw_unpack_tensor.mutable_data<int8_t>(cpu_place);
  cpu_int4_quanted_weight_raw_unpack(out_cpu_tensor, &raw_unpack_tensor);

  // transpose
  DenseTensor transposed_tensor;
  transposed_tensor.Resize(
      {raw_unpack_tensor.dims()[1], raw_unpack_tensor.dims()[0]});
  transposed_tensor.mutable_data<int8_t>(cpu_place);
  cpu_2d_tensor_transpose(raw_unpack_tensor, &transposed_tensor);

  // col pack
  out_cpu_tensor.Resize(
      {transposed_tensor.dims()[0], transposed_tensor.dims()[1] / 2});
  cpu_int4_quanted_weight_col_pack(transposed_tensor, &out_cpu_tensor);

  out_cpu_tensor.Resize({n / 2, m});
  out->Resize({n / 2, m});
  phi::Copy(dev_ctx, out_cpu_tensor, dev_ctx.GetPlace(), true, out);
}

__global__ void int4_quanted_matrix_raw_unpack_kernel(const int8_t* mat,
                                                      int8_t* unpack_mat,
                                                      int M,
                                                      int N) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int i = global_idx / N;
  int j = global_idx % N;

  if (global_idx >= M * N) {
    return;
  }

  int8_t val = mat[global_idx];
  int8_t low = val & 0x0F;
  int8_t mask = ((low & 0x80) == 0) & ((low & 0x78) != 0);
  low -= 16 * mask;

  int8_t high = (val >> 4) & 0x0F;
  mask = ((high & 0x80) == 0) & ((high & 0x78) != 0);
  high -= 16 * mask;

  int output_global_idx0 = (2 * i) * N + j;
  int output_global_idx1 = (2 * i + 1) * N + j;

  unpack_mat[output_global_idx0] = low;
  unpack_mat[output_global_idx1] = high;
}

__global__ void int4_quanted_matrix_col_pack_kernel(const int8_t* mat,
                                                    int8_t* pack_mat,
                                                    int M,
                                                    int N) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int i = global_idx / N;
  int j = global_idx % N;

  if (global_idx >= M * N) {
    return;
  }

  int mat_global_idx0 = i * 2 * N + 2 * j;
  int mat_global_idx1 = i * 2 * N + 2 * j + 1;

  int8_t low = mat[mat_global_idx0] & 0x0F;
  low = low + ((low >> 3) & 1) * 16;

  int8_t high = mat[mat_global_idx1] & 0x0F;
  high = high + ((high >> 3) & 1) * 16;

  pack_mat[global_idx] = ((high << 4) & 0xf0) | (low & 0x0f);
}

void gpu_int4_quantized_weight_layout_trans_impl(
    const CustomContext& dev_ctx,
    const std::vector<int64_t>& shape,
    DenseTensor* out) {
  int64_t total_m = shape[0];
  int64_t total_n = shape[1];
  out->Resize({total_m / 2, total_n});

  DenseTensor unpack_mat(out->type());
  unpack_mat.Resize({total_m, total_n});
  dev_ctx.template Alloc<int8_t>(&unpack_mat);

  constexpr int kBlockSize = 64;
  int64_t kGridSize = (out->numel() + kBlockSize - 1) / kBlockSize;
  int4_quanted_matrix_raw_unpack_kernel<<<kGridSize, kBlockSize>>>(
      out->data<int8_t>(),
      unpack_mat.data<int8_t>(),
      out->dims()[0],
      out->dims()[1]);

  DenseTensor transposed_tensor;
  transposed_tensor.Resize({unpack_mat.dims()[1], unpack_mat.dims()[0]});
  dev_ctx.template Alloc<int8_t>(&transposed_tensor);
  std::vector<int> axis = {1, 0};
  funcs::Transpose<CustomContext, int8_t, 2> trans;
  trans(dev_ctx, unpack_mat, &transposed_tensor, axis);

  out->Resize({transposed_tensor.dims()[0], transposed_tensor.dims()[1] / 2});
  int4_quanted_matrix_col_pack_kernel<<<kGridSize, kBlockSize>>>(
      transposed_tensor.data<int8_t>(),
      out->data<int8_t>(),
      out->dims()[0],
      out->dims()[1]);

  out->Resize({total_n / 2, total_m});
}

template <typename Context>
void MetaxQuantizedWeightLayoutTrans(const Context& dev_ctx,
                                     const std::string& algo,
                                     const std::vector<int64_t>& shape,
                                     DenseTensor* out) {
  if (algo == "weight_only_int4") {
    if (dev_ctx.GetPlace() == phi::CPUPlace()) {
      cpu_int4_quantized_weight_layout_trans_impl(dev_ctx, shape, out);
    } else {
      gpu_int4_quantized_weight_layout_trans_impl(dev_ctx, shape, out);
    }

  } else {
    PADDLE_FATAL(
        "The algo must be in ['weight_only_int4'"
        "], but got[%s]",
        algo);
  }
}

}  // namespace phi
