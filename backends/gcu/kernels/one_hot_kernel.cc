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

#include "common/common.h"
#include "common/utils.h"
#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_name_list.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {
template <typename T, typename Context>
void OneHotKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& num_classes_s,
                  phi::DenseTensor* out) {
  int depth = num_classes_s.to<int>();
  auto out_dims = out->dims();
  if (out_dims.size() > 0 && out_dims[out_dims.size() - 1] == -1) {
    out_dims[out_dims.size() - 1] = depth;
    out->Resize(out_dims);
  }
  dev_ctx.template Alloc<float>(out);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "one_hot_v2", one_hot_v2);

    auto x_int32 = x;
    if (x.dtype() == phi::DataType::INT64)
      x_int32 = cast(dev_ctx, x, phi::DataType::INT32);

    one_hot(dev_ctx, x_int32, x_int32.dims().size(), depth, *out);

    PADDLE_GCU_KERNEL_END("one_hot_v2", one_hot_v2);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["depth"] = depth;
    attrs["dtype"] = static_cast<int>(phi::DataType::FLOAT32);

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "one_hot_v2",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    one_hot, gcu, ALL_LAYOUT, custom_kernel::OneHotKernel, int32_t, int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
}
