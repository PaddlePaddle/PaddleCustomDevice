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

constexpr int64_t kNoPadding = -1;

template <typename T, typename Context>
void EmbeddingKernel(const Context& dev_ctx,
                     const phi::DenseTensor& inputx,
                     const phi::DenseTensor& weight,
                     int64_t padding_idx,
                     phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();

  dev_ctx.template Alloc<T>(out);

  if (padding_idx == kNoPadding) {
    NpuOpRunner runner;
    runner.SetType("GatherV2")
        .AddInput(weight)
        .AddInput(inputx)
        .AddInput(dev_ctx, std::vector<int32_t>{0})
        .AddAttrs({{"batch_dims", 0}})
        .AddOutput(*out);
    runner.Run(stream);
  } else {
    phi::DenseTensor tmp_table_t;
    phi::DenseTensorMeta table_meta = {weight.dtype(), weight.dims()};
    tmp_table_t.set_meta(table_meta);
    dev_ctx.template Alloc<T>(&tmp_table_t);

    phi::DenseTensor index;
    phi::DenseTensorMeta index_meta = {phi::DataType::FLOAT32, {1, 1}};
    index.set_meta(index_meta);
    dev_ctx.template Alloc<T>(&index);

    FillNpuTensorWithConstant<int32_t>(
        &index, dev_ctx, static_cast<int32_t>(padding_idx));
    index.Resize({1, 1});

    auto updata_dim = phi::make_ddim({1, weight.dims()[1]});
    phi::DenseTensor update;
    update.Resize(updata_dim);
    dev_ctx.template Alloc<T>(&update);

    FillNpuTensorWithConstant<T>(&update, dev_ctx, static_cast<T>(0));
    update.Resize(updata_dim);

    NpuOpRunner update_runner;
    update_runner.SetType("TensorScatterUpdate")
        .AddInput(weight)
        .AddInput(index)
        .AddInput(update)
        .AddOutput(tmp_table_t);
    update_runner.Run(stream);

    NpuOpRunner runner;
    runner.SetType("GatherV2")
        .AddInput(tmp_table_t)
        .AddInput(inputx)
        .AddInput(dev_ctx, std::vector<int32_t>{0})
        .AddAttrs({{"batch_dims", 0}})
        .AddOutput(*out);
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void EmbeddingGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& input,
                         const phi::DenseTensor& weight,
                         const phi::DenseTensor& out_grad,
                         int64_t padding_idx,
                         phi::DenseTensor* weight_grad) {
  dev_ctx.template Alloc<T>(weight_grad);

  auto stream = dev_ctx.stream();

  const auto& runner_zeros =
      NpuOpRunner("ZerosLike", {*weight_grad}, {*weight_grad});
  runner_zeros.Run(stream);

  if (padding_idx == kNoPadding) {
    // NOTE(zhiqiu): It seems in cann 20.1, the first input and output
    // can be different tensor, but in cann 20.2+, it does inplace operation.
    // Thus, the first input and output should be same tensor.
    const auto& runner_scatter = NpuOpRunner("ScatterAdd",
                                             {*weight_grad, input, out_grad},
                                             {*weight_grad},
                                             {{"use_locking", true}});
    runner_scatter.Run(stream);
  } else {
    phi::DenseTensor casted_inputx;
    if (input.dtype() != phi::DataType::INT32) {
      phi::DenseTensorMeta meta = {phi::DataType::INT32, input.dims()};
      casted_inputx.set_meta(meta);
      dev_ctx.template Alloc<int32_t>(&casted_inputx);
      const auto& cast_runner = NpuOpRunner(
          "Cast", {input}, {casted_inputx}, {{"dst_type", ACL_INT32}});
      cast_runner.Run(stream);
    } else {
      casted_inputx = input;
    }
    auto table_grad_dims = weight_grad->dims();

    NpuOpRunner runner;
    runner.SetType("UnsortedSegmentSum")
        .AddInput(out_grad)
        .AddInput(casted_inputx)
        .AddInput(dev_ctx, std::vector<int64_t>{table_grad_dims[0]})
        .AddOutput(*weight_grad);
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(embedding,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::EmbeddingKernel,
                          float,
                          int,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(embedding_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::EmbeddingGradKernel,
                          float,
                          int,
                          phi::dtype::float16) {}
