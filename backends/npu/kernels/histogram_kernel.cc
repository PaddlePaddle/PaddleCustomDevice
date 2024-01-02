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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
namespace custom_kernel {
template <typename T, typename Context>
void HistogramKernel(const Context& dev_ctx,
                     const phi::DenseTensor& input,
                     int64_t bins,
                     int min,
                     int max,
                     phi::DenseTensor* output) {
  int mbins = static_cast<int>(bins);
  dev_ctx.template Alloc<int>(output);
  phi::DenseTensor ranges;
  phi::DenseTensor nbins;
  T mmin = static_cast<T>(min);
  T mmax = static_cast<T>(max);
  std::vector<T> mrange{mmax, mmin};
  TensorFromVector<T>(dev_ctx, mrange, dev_ctx, &ranges);
  std::vector<T> ss = {bins};
  TensorFromVector<T>(dev_ctx, ss, dev_ctx, &nbins);
  auto output_dim = output->dims();
  output->Resize({-1});
  NpuOpRunner histogram_runner;
  histogram_runner.SetType("HistogramFixedWidth")
      .AddInput(input)
      .AddInput(ranges)
      .AddInput(nbins)
      .AddOutput(*output)
      .Run(dev_ctx.stream());
  output->Resize(output_dim);
}
};  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(histogram,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HistogramKernel,
                          float,
                          int,
                          int64_t) {
  kernel->OutputAt(0).SetDataType(paddle::DataType::INT64);
}
