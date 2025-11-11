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

inline int ComputeStride(int axis, phi::DDim dims) {
  int size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T, typename Context>
void DiagKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                int offset,
                float padding_value,
                phi::DenseTensor* out) {
  auto* x_data = x.data<T>();
  auto x_dims = x.dims();
  dev_ctx.template Alloc<T>(out);
  auto out_dims = out->dims();
  if (x_dims.size() == 1) {
    if (offset == 0 && abs(padding_value) < 1e-6) {
      auto stream = dev_ctx.stream();
      const auto& runner = NpuOpRunner("Diag", {x}, {*out}, {});
      runner.Run(stream);
    } else {
      phi::DenseTensor cpu_out_tensor;
      cpu_out_tensor.Resize(out_dims);
      auto cpu_out_data = dev_ctx.template HostAlloc<T>(&cpu_out_tensor);
      int64_t output_numel = phi::product(out_dims);
      for (int i = 0; i < output_numel; ++i) {
        cpu_out_data[i] = padding_value;
      }
      auto x_length = x_dims[0];
      const int& x_stride = ComputeStride(0, x_dims);

      auto out_stride_0 = ComputeStride(0, out_dims);
      auto out_stride_1 = ComputeStride(1, out_dims);
      cpu_out_data +=
          (offset >= 0 ? offset * out_stride_1 : -offset * out_stride_0);
      std::vector<T> x_data_vec;
      TensorToVector(dev_ctx, *&x, dev_ctx, &x_data_vec);

      for (int i = 0; i < x_length; i++) {
        cpu_out_data[i * (out_stride_0 + out_stride_1)] =
            x_data_vec[i * x_stride];
      }
      TensorCopy(dev_ctx, cpu_out_tensor, true, out);
    }
  } else {
    phi::DenseTensor cpu_out_tensor;
    cpu_out_tensor.Resize(out_dims);
    auto cpu_out_data = dev_ctx.template HostAlloc<T>(&cpu_out_tensor);
    auto out_length = out_dims[0];
    const int& x_stride_0 = ComputeStride(0, x_dims);
    const int& x_stride_1 = ComputeStride(1, x_dims);

    auto out_stride_0 = ComputeStride(0, out_dims);
    std::vector<T> x_data_vec;
    TensorToVector(dev_ctx, *&x, dev_ctx, &x_data_vec);
    auto x_data_offset =
        offset >= 0 ? offset * x_stride_1 : -offset * x_stride_0;

    for (int i = 0; i < out_length; i++) {
      cpu_out_data[i * out_stride_0] =
          x_data_vec[x_data_offset + i * (x_stride_0 + x_stride_1)];
    }
    TensorCopy(dev_ctx, cpu_out_tensor, true, out);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(diag,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DiagKernel,
                          phi::dtype::float16,
                          int,
                          float,
                          double,
                          int64_t) {}
