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

#include "kernels/cast_kernel.h"
#include "kernels/funcs/gcu_funcs.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/funcs/embedding_util.h"

namespace custom_kernel {
template <typename T, typename Context>
void EmbeddingKernel(const Context& dev_ctx,
                     const phi::DenseTensor& inputx,
                     const phi::DenseTensor& weight,
                     int64_t padding_idx,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

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

template <typename T, typename Context>
void EmbeddingGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& input,
                         const phi::DenseTensor& weight,
                         const phi::DenseTensor& out_grad,
                         int64_t padding_idx,
                         phi::DenseTensor* weight_grad) {
  dev_ctx.template Alloc<T>(weight_grad);

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

template <typename T, typename Context>
void EmbeddingSparseGradKernel(const Context& ctx,
                               const DenseTensor& input,
                               const DenseTensor& weight,
                               const DenseTensor& out_grad,
                               int64_t padding_idx,
                               phi::SelectedRows* weight_grad) {
  phi::DDim table_dim = weight.dims();
  auto gcu_place = ctx.GetPlace();

  std::vector<int64_t> ids;
  phi::DenseTensor ids_cpu;
  ids_cpu.Resize(input.dims());
  ctx.template HostAlloc(
      &ids_cpu, phi::DataType::INT64, input.numel() * sizeof(int64_t));
  if (input.dtype() == phi::DataType::INT64) {
    // phi::Copy(ctx, input, phi::CPUPlace(), false, &ids_cpu);

    TensorCopy(ctx, input, false, &ids_cpu, phi::CPUPlace());

    ids = phi::CopyIdsToVector<int64_t, int64_t>(ids_cpu);
  } else if (input.dtype() == phi::DataType::INT32) {
    // cast
    DenseTensor tmp_out = DenseTensor();
    tmp_out.Resize(input.dims());
    ctx.template Alloc<int64_t>(&tmp_out);
    GcuCastKernel<int32_t, Context>(ctx, input, phi::DataType::INT64, &tmp_out);

    TensorCopy(ctx, tmp_out, false, &ids_cpu, phi::CPUPlace());

    // phi::memory_utils::Copy(phi::CPUPlace(),
    //                         ids_cpu.data(),
    //                         input.place(),
    //                         tmp_out.data(),
    //                         sizeof(int64_t) * input.numel());
    ids = phi::CopyIdsToVector<int, int64_t>(ids_cpu);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "emebdding input only support int32 and int64"));
  }

  auto ids_num = static_cast<int64_t>(input.numel());
  // Since paddings are not trainable and fixed in forward, the gradient of
  // paddings makes no sense and we don't deal with it in backward.
  auto* d_table = weight_grad;
  auto* d_output = &out_grad;
  d_table->set_rows(ids);

  auto* d_table_value = d_table->mutable_value();
  d_table_value->Resize({ids_num, table_dim[1]});
  ctx.template Alloc<T>(d_table_value);

  d_table->set_height(table_dim[0]);

  auto* d_output_data = d_output->template data<T>();
  auto* d_table_data = d_table_value->template data<T>();

  auto d_output_dims = d_output->dims();
  auto d_output_dims_2d =
      flatten_to_2d(d_output_dims, d_output_dims.size() - 1);
  PADDLE_ENFORCE_EQ(d_table_value->dims(),
                    d_output_dims_2d,
                    phi::errors::InvalidArgument(
                        "ShapeError: The shape of lookup_table@Grad and "
                        "output@Grad should be same. "
                        "But received lookup_table@Grad's shape = [%s], "
                        "output@Grad's shape = [%s].",
                        d_table_value->dims(),
                        d_output_dims_2d));

  phi::memory_utils::Copy(gcu_place,
                          d_table_data,
                          gcu_place,
                          d_output_data,
                          d_output->numel() * sizeof(T));
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
// PD_REGISTER_PLUGIN_KERNEL(embedding_sparse_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::EmbeddingSparseGradKernel,
//                           float,
//                           int,
//                           phi::dtype::float16) {}
