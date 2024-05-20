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
void TopkKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& k_scalar,
                int axis,
                bool largest,
                bool sorted,
                phi::DenseTensor* out,
                phi::DenseTensor* indices) {
  PADDLE_GCU_KERNEL_TRACE("topk");
  if (axis < 0) {
    axis += x.dims().size();
  }

  int k = k_scalar.to<int>();

  phi::DDim output_dims = x.dims();
  output_dims[axis] = k;

  out->Resize(output_dims);
  indices->Resize(output_dims);

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<int64_t>(indices);

  // Support 0D
  if (output_dims.size() == 0) {
    TensorCopy(dev_ctx, x, false, out);
    FillGcuTensorWithConstant<int64_t>(
        indices, dev_ctx, static_cast<int64_t>(0L));
    // dev_ctx.Wait();
    return;
  }

  if (LaunchAOTKernel()) {
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor output_value =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);
    auto meta = indices->meta();
    meta.dtype = phi::DataType::INT32;
    phi::DenseTensor output_indices = TensorEmpty(dev_ctx, meta);

    LAUNCH_TOPSATENOP(topsatenTopk,
                      dev_ctx,
                      output_value,
                      output_indices,
                      input_x,
                      k,
                      axis,
                      sorted,
                      largest);

    MaybeTransResult(dev_ctx, output_value, out);
    custom_kernel::Cast(dev_ctx, output_indices, phi::DataType::INT64, indices);

  } else {  // kernel impl base on JIT
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
