// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T>
std::vector<T> ComputeDimStride(const std::vector<T> dim) {
  size_t dim_size = dim.size();
  std::vector<T> dim_strides;
  dim_strides.resize(dim_size);
  for (size_t i = 0; i < dim_size - 1; i++) {
    size_t temp_stride = 1;
    for (size_t j = i + 1; j < dim_size; j++) {
      temp_stride = temp_stride * dim[j];
    }
    dim_strides[i] = temp_stride;
  }
  dim_strides[dim_size - 1] = 1;
  return dim_strides;
}

template <typename T, typename Context>
void DiagonalKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    int offset,
                    int axis1,
                    int axis2,
                    phi::DenseTensor* out) {
  auto input_dim = phi::vectorize(x.dims());
  auto input_dim_size = input_dim.size();
  auto output_dim = phi::vectorize(out->dims());
  auto output_dim_size = output_dim.size();

  // copy to cpu
  phi::DenseTensor output_cpu_t;
  output_cpu_t.Resize(out->dims());
  auto output_data = dev_ctx.template HostAlloc<T>(&output_cpu_t);

  std::vector<T> input_data;
  TensorToVector(dev_ctx, *&x, dev_ctx, &input_data);

  const int64_t offset_ = offset;
  int64_t axis1_ = axis1 < 0 ? input_dim_size + axis1 : axis1;
  int64_t axis2_ = axis2 < 0 ? input_dim_size + axis2 : axis2;

  std::vector<int64_t> input_stride = ComputeDimStride(input_dim);
  std::vector<int64_t> output_stride = ComputeDimStride(output_dim);

  int64_t out_numel = out->numel();
  for (int64_t idx = 0; idx < out_numel; idx++) {
    std::vector<int64_t> idx_dim(output_dim_size);
    int64_t temp = 0;
    for (size_t i = 0; i < output_dim_size; i++) {
      idx_dim[i] = (idx - temp) / output_stride[i];
      temp = temp + idx_dim[i] * output_stride[i];
    }
    int64_t tmp = idx_dim[output_dim_size - 1];
    std::vector<int64_t> list;
    list.clear();
    int64_t l = std::min(axis1_, axis2_);
    int64_t r = std::max(axis1_, axis2_);
    for (size_t j = 0; j < output_dim_size - 1; j++) {
      list.push_back(idx_dim[j]);
    }
    if (offset_ == 0) {
      list.insert(list.begin() + l, tmp);
      list.insert(list.begin() + r, tmp);
    } else if (offset_ > 0) {
      if (axis1_ < axis2_) {
        list.insert(list.begin() + l, tmp);
        list.insert(list.begin() + r, tmp + offset_);
      } else {
        list.insert(list.begin() + l, tmp + offset_);
        list.insert(list.begin() + r, tmp);
      }
    } else if (offset_ < 0) {
      if (axis1_ < axis2_) {
        list.insert(list.begin() + l, tmp - offset_);
        list.insert(list.begin() + r, tmp);
      } else {
        list.insert(list.begin() + l, tmp);
        list.insert(list.begin() + r, tmp - offset_);
      }
    }

    int64_t input_offset = 0;
    for (size_t i = 0; i < input_dim_size; i++) {
      input_offset = input_offset + list[i] * input_stride[i];
    }
    output_data[idx] = input_data[input_offset];
  }
  TensorCopy(dev_ctx, output_cpu_t, true, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    diagonal, npu, ALL_LAYOUT, custom_kernel::DiagonalKernel, float) {}
