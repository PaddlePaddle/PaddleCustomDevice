// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();

  phi::DenseTensorMeta meta = input.meta();
  meta.strides = meta.calc_strides(meta.dims);
  meta.offset = 0;
  out->set_meta(meta);
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor offset_tensor;
  offset_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&offset_tensor);
  FillNpuTensorWithConstant<int64_t>(
      &offset_tensor, dev_ctx, static_cast<int64_t>(0));

  NpuOpRunner runner;
  runner.SetType("AsStrided")
      .AddInput(input)
      .AddInput(dev_ctx, std::move(phi::vectorize(input.dims())))
      .AddInput(dev_ctx, std::move(phi::vectorize(input.strides())))
      .AddInput(offset_tensor)
      .AddOutput(*out)
      .Run(stream);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(contiguous,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ContiguousKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
