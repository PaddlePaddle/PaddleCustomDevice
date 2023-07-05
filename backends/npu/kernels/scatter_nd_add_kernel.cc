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
void ScatterNdAddKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& index,
                        const phi::DenseTensor& updates,
                        phi::DenseTensor* out) {
  const auto& index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s] or [%s].",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  if (index.numel() != 0) {
    // NOTE(songkai05): ScatterNdAdd updates input inplace
    TensorCopy(dev_ctx, x, true, out);

    NpuOpRunner runner;
    runner.SetType("ScatterNdAdd")
        .AddInput(*out)
        .AddInput(index)
        .AddInput(updates)
        .AddOutput(*out)
        .AddAttr("use_locking", false);
    runner.Run(stream);
  } else {
    // deal with the situation when index is empty
    std::vector<int> axes;
    int dims_size = x.dims().size();
    int update_size = updates.dims().size();
    for (size_t i = 0; i < update_size - dims_size; ++i) {
      axes.push_back(i);
    }

    phi::DenseTensor updates_sum;
    updates_sum.Resize(x.dims());
    dev_ctx.template Alloc<T>(&updates_sum);

    NpuOpRunner runner1;
    runner1.SetType("ReduceSum")
        .AddInput(updates)
        .AddInput(dev_ctx, std::move(axes))
        .AddOutput(updates_sum)
        .AddAttr("keep_dims", false);
    runner1.Run(stream);

    NpuOpRunner runner2;
    runner2.SetType("Add").AddInput(x).AddInput(updates_sum).AddOutput(*out);
    runner2.Run(stream);
  }
}

template <typename T, typename Context>
void ScatterNdAddGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& index,
                            const phi::DenseTensor& updates UNUSED,
                            const phi::DenseTensor& out_grad,
                            phi::DenseTensor* x_grad,
                            phi::DenseTensor* updates_grad) {
  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
    TensorCopy(dev_ctx, out_grad, true, x_grad);
  }
  if (updates_grad) {
    dev_ctx.template Alloc<T>(updates_grad);
    const auto& runner =
        NpuOpRunner("GatherNd", {out_grad, index}, {*updates_grad}, {});
    runner.Run(dev_ctx.stream());
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scatter_nd_add,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ScatterNdAddKernel,
                          phi::dtype::float16,
                          float,
                          int) {}

PD_REGISTER_PLUGIN_KERNEL(scatter_nd_add_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ScatterNdAddGradKernel,
                          phi::dtype::float16,
                          float,
                          int) {}
