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
void MaskedSelectKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& mask,
                        phi::DenseTensor* out) {
  auto input_dim = x.dims();
  auto mask_dim = mask.dims();
  PADDLE_ENFORCE_EQ(input_dim,
                    mask_dim,
                    phi::errors::InvalidArgument(
                        "The dim size of input and mask in OP(masked_selected) "
                        "must be equal, but got input dim:(%ld), mask dim: "
                        "(%ld). Please check input "
                        "value.",
                        input_dim,
                        mask_dim));

  auto stream = dev_ctx.stream();

  phi::DenseTensor mask_int32;
  phi::DenseTensor out_size;
  mask_int32.Resize(mask_dim);
  out_size.Resize({1});
  dev_ctx.template Alloc<int32_t>(&mask_int32);
  dev_ctx.template Alloc<int32_t>(&out_size);
  std::vector<int32_t> out_size_vec;
  {
    const auto& cast_runner = NpuOpRunner(
        "Cast",
        {mask},
        {mask_int32},
        {{"dst_type",
          static_cast<int32_t>(ConvertToNpuDtype(phi::DataType::INT32))}});
    cast_runner.Run(stream);

    mask_int32.Resize({mask_int32.numel()});

    NpuOpRunner sum_runner;
    sum_runner.SetType("ReduceSum");
    sum_runner.AddInput(mask_int32);
    sum_runner.AddInput(dev_ctx, std::vector<int32_t>(1, 0));
    sum_runner.AddOutput(out_size);
    sum_runner.AddAttr("keep_dims", false);
    sum_runner.Run(stream);

    // wait for ReduceSum complete
    dev_ctx.Wait();
    TensorToVector(dev_ctx, out_size, dev_ctx, &out_size_vec);
    // wait for copy complete
    dev_ctx.Wait();
  }

  out->Resize(phi::make_ddim({out_size_vec[0]}));
  dev_ctx.template Alloc<T>(out);

  const auto& runner = NpuOpRunner("MaskedSelect", {x, mask}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void MaskedSelectGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& mask,
                            const phi::DenseTensor& out_grad,
                            phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  auto stream = dev_ctx.stream();

  phi::DenseTensor mask_int32;
  phi::DenseTensor out_size;
  mask_int32.Resize(mask.dims());
  out_size.Resize({1});
  dev_ctx.template Alloc<int32_t>(&mask_int32);
  dev_ctx.template Alloc<int32_t>(&out_size);

  std::vector<int32_t> out_size_vec(1, out_grad.numel());
  TensorFromVector(dev_ctx, out_size_vec, dev_ctx, &out_size);
  {
    const auto& cast_runner = NpuOpRunner(
        "Cast",
        {mask},
        {mask_int32},
        {{"dst_type",
          static_cast<int32_t>(ConvertToNpuDtype(phi::DataType::INT32))}});
    cast_runner.Run(stream);

    mask_int32.Resize({mask_int32.numel()});
  }

  phi::DenseTensor topkv2_out;
  phi::DenseTensor indices;
  topkv2_out.Resize({out_size_vec[0]});
  indices.Resize({out_size_vec[0]});
  dev_ctx.template Alloc<int32_t>(&topkv2_out);
  dev_ctx.template Alloc<int32_t>(&indices);
  {
    NpuOpRunner topkv2_runner;
    topkv2_runner.SetType("TopKV2")
        .AddInput(mask_int32)
        .AddInput(out_size)
        .AddOutput(topkv2_out)
        .AddOutput(indices)
        .AddAttr("sorted", false)
        .AddAttr("dim", 0)
        .AddAttr("largest", true)
        .Run(stream);

    NpuOpRunner topkv2_runner2;
    topkv2_runner2.SetType("TopKV2")
        .AddInput(indices)
        .AddInput(out_size)
        .AddOutput(topkv2_out)
        .AddOutput(indices)
        .AddAttr("sorted", true)
        .AddAttr("dim", 0)
        .AddAttr("largest", false)
        .Run(stream);

    topkv2_out.Resize({out_size_vec[0], 1});
    x_grad->Resize({x_grad->numel()});

    NpuOpRunner scatter_runner;
    scatter_runner.SetType("ScatterNd");
    scatter_runner.AddInput(topkv2_out);
    scatter_runner.AddInput(out_grad);
    scatter_runner.AddInput(
        dev_ctx,
        std::vector<int32_t>(1, static_cast<int32_t>(x_grad->numel())));
    scatter_runner.AddOutput(*x_grad);
    scatter_runner.Run(stream);

    x_grad->Resize(mask.dims());
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(masked_select,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MaskedSelectKernel,
                          phi::dtype::float16,
                          float,
                          int,
                          int64_t) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(masked_select_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MaskedSelectGradKernel,
                          phi::dtype::float16,
                          float,
                          int,
                          int64_t) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
