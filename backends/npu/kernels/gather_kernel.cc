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
void GatherKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& index,
                  const phi::Scalar& axis,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const auto& runner =
      NpuOpRunner("Gather", {x, index}, {*out}, {{"validate_indices", true}});
  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void GatherGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& index,
                      const phi::DenseTensor& out_grad,
                      const phi::Scalar& axis,
                      bool overwrite,
                      phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  const phi::DenseTensor* p_index = &index;
  // step1: Unsqueeze index
  phi::DenseTensor tmp_tensor(index);
  const auto index_dims = index.dims();
  if (index_dims.size() == 1) {
    std::vector<int64_t> new_dim = {index_dims[0], 1};
    tmp_tensor.Resize(phi::make_ddim(new_dim));
    p_index = &tmp_tensor;
  }

  auto stream = dev_ctx.stream();

  // step2: ZerosLike x in device
  phi::DenseTensor zeroslike_xout;
  phi::DenseTensorMeta meta = {x_grad->dtype(), x.dims()};
  zeroslike_xout.set_meta(meta);
  dev_ctx.template Alloc<T>(&zeroslike_xout);

  FillNpuTensorWithConstant<T>(&zeroslike_xout, dev_ctx, static_cast<T>(0));
  zeroslike_xout.Resize(x.dims());

  // step3: scatter(x_grad)
  const auto& runner_scatter = NpuOpRunner("TensorScatterUpdate",
                                           {zeroslike_xout, *p_index, out_grad},
                                           {*x_grad},
                                           {});
  runner_scatter.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gather,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::GatherKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gather_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::GatherGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
