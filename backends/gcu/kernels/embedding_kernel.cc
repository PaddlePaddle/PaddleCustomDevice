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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
template <typename T, typename Context>
void EmbeddingKernel(const Context& dev_ctx,
                     const phi::DenseTensor& inputx,
                     const phi::DenseTensor& weight,
                     int64_t padding_idx,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("embedding");
  PADDLE_ENFORCE_EQ(
      padding_idx,
      -1,
      phi::errors::InvalidArgument(
          "padding_idx only support none, but got %ld.", padding_idx));
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    phi::DenseTensor x = MaybeCreateOrTrans64To32bits(dev_ctx, inputx);
    LAUNCH_TOPSATENOP(
        topsatenEmbedding, dev_ctx, *out, weight, x, padding_idx, false, false);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Ids"] = {"inputx"};
    input_names["W"] = {"weight"};

    TensorValueMap inputs;
    inputs["Ids"] = {const_cast<DenseTensor*>(&inputx)};
    inputs["W"] = {const_cast<DenseTensor*>(&weight)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["padding_idx"] = padding_idx;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "lookup_table_v2",
              dev_ctx);
  }
}

template <typename T, typename Context>
void EmbeddingGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& input,
                         const phi::DenseTensor& weight,
                         const phi::DenseTensor& out_grad,
                         int64_t padding_idx,
                         phi::DenseTensor* weight_grad) {
  PADDLE_GCU_KERNEL_TRACE("embedding_grad");
  dev_ctx.template Alloc<T>(weight_grad);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Ids"] = {"input"};
    input_names["W"] = {"weight"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["Ids"] = {const_cast<DenseTensor*>(&input)};
    inputs["W"] = {const_cast<DenseTensor*>(&weight)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

    TensorNameMap output_names;
    output_names[GradVarName("Weight")] = {"weight_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("Weight")] = {weight_grad};

    GcuAttributeMap attrs;
    attrs["padding_idx"] = padding_idx;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "lookup_table_v2_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(embedding,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::EmbeddingKernel,
                          float,
                          int,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(embedding_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::EmbeddingGradKernel,
                          float,
                          int,
                          phi::dtype::float16) {}
