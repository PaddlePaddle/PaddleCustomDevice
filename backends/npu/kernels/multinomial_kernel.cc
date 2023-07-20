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

template <typename T, typename Context>
void MultinomialKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::Scalar& num,
                       bool replacement,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<int64_t>(out);
  auto num_samples = num.to<int>();
  auto stream = dev_ctx.stream();

  auto& engine = *dev_ctx.GetGenerator()->GetCPUEngine();
  auto seed_val = static_cast<int>(engine());

  phi::DenseTensor seed, offset;
  seed.Resize({1});
  offset.Resize({1});
  FillNpuTensorWithConstant<int64_t>(&seed, dev_ctx, seed_val);
  FillNpuTensorWithConstant<int64_t>(&offset, dev_ctx, 0);

  NpuOpRunner runner;
  runner.SetType("MultinomialWithReplacement")
      .AddInput(x)
      .AddInput(seed)
      .AddInput(offset)
      .AddOutput(*out)
      .AddAttr("numsamples", num_samples)
      .AddAttr("replacement", replacement);
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(multinomial,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MultinomialKernel,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
