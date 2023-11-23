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

#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void TopkKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& k_scalar,
                int axis,
                bool largest,
                bool sorted,
                phi::DenseTensor* out,
                phi::DenseTensor* indices) {
  if (axis < 0) {
    axis += x.dims().size();
  }

  int k = k_scalar.to<int>();

  phi::DDim output_dims = x.dims();
  output_dims[axis] = k;

  out->Resize(output_dims);
  indices->Resize(output_dims);

  // Support 0D
  if (output_dims.size() == 0) {
    dev_ctx.template Alloc<T>(out);
    dev_ctx.template Alloc<int64_t>(indices);
    TensorCopy(dev_ctx, x, true, out);
    FillGcuTensorWithConstant<int64_t>(
        indices, dev_ctx, static_cast<int64_t>(0.0));
    indices->Resize(output_dims);
    dev_ctx.Wait();
    return;
  }

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "topk", topk);
    dev_ctx.template Alloc<int32_t>(indices);

    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_out = *out;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
      dev_ctx.template Alloc<int32_t>(out);
    } else {
      dev_ctx.template Alloc<T>(out);
    }

    auto x_gcu = GetHlirTensor(tmp_x);
    auto out_gcu = GetHlirTensor(*out);
    auto indices_gcu = GetHlirTensor(*indices);
    hlir::DispatchParam params;
    params.inputs = {x_gcu};
    params.outputs = {out_gcu, indices_gcu};
    params.metadata.setValue("k", static_cast<int64_t>(k));
    params.metadata.setValue("axis", static_cast<int64_t>(axis));
    params.metadata.setValue("is_sorted", sorted);
    params.metadata.setValue("cmp_mod", static_cast<int64_t>(largest ? 1 : 2));
    params.metadata.setValue("stable_mod", static_cast<int64_t>(1));
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());

    AOTOPS_DEBUG(kTopk, params);
    GCUOPS_TRACE_START(relu_grad);
    auto func_ptr = GetOpFuncPtr(kTopk, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass, phi::errors::InvalidArgument("dispatch %s failed!", kTopk));
    } else {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("not find aot func for %s", kTopk));
    }
    FreeDispatchParam(params);
    GCUOPS_TRACE_END(relu_grad);
    GcuOpStreamSync(dev_ctx);
    PADDLE_GCU_KERNEL_END("topk", topk);

    if (x.dtype() == phi::DataType::INT64) {
      *out = cast(dev_ctx, *out, phi::DataType::INT64);
    }
    *indices = cast(dev_ctx, *indices, phi::DataType::INT64);
  } else {
    dev_ctx.template Alloc<T>(out);
    dev_ctx.template Alloc<int64_t>(indices);
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};
    output_names["Indices"] = {"indices"};

    TensorValueMap outputs;
    outputs["Out"] = {out};
    outputs["Indices"] = {indices};

    GcuAttributeMap attrs;
    attrs["k"] = k;
    attrs["axis"] = axis;
    attrs["largest"] = largest;
    attrs["sorted"] = sorted;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "top_k_v2", dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(topk,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TopkKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int,
                          int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
