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

#define MAX_RANK_SUPPORTED 6

template <typename T, typename Context>
void ExpandAsKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const paddle::optional<phi::DenseTensor>& y,
                    const std::vector<int>& target_shape,
                    phi::DenseTensor* out) {
  auto rank = x.dims().size();
  auto target_rank = target_shape.size();
  PADDLE_ENFORCE_GE(target_rank,
                    rank,
                    phi::errors::InvalidArgument(
                        "The rank (%d) of the input 'target_tensor' for "
                        "expand_as_v2 op must be greater than or equal to "
                        "the rank (%d) of the input 'x'.",
                        target_rank,
                        rank));
  PADDLE_ENFORCE_GE(
      rank,
      0,
      phi::errors::InvalidArgument("The rank (%d) of the input 'x' for "
                                   "expand_as_v2 op must be positive.",
                                   rank));
  PADDLE_ENFORCE_LE(target_rank,
                    MAX_RANK_SUPPORTED,
                    phi::errors::InvalidArgument(
                        "The rank (%d) of the input 'target_tensor' for "
                        "expand_as_v2 op must be less than or equal to %d.",
                        target_rank,
                        MAX_RANK_SUPPORTED));

  auto in_dims = x.dims();
  auto vec_in_dims = phi::vectorize<int>(in_dims);
  auto diff = target_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);

  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    PADDLE_ENFORCE_NE(target_shape[i],
                      0,
                      phi::errors::InvalidArgument(
                          "The value of target shape cannot be zero."));
    if (vec_in_dims[i] != 1) {
      PADDLE_ENFORCE_EQ(
          vec_in_dims[i],
          target_shape[i],
          phi::errors::InvalidArgument(
              "The value (%d) of the non-singleton dimension does not match"
              " the corresponding value (%d) in "
              "target tensor for expand_as_v2 op.",
              vec_in_dims[i],
              target_shape[i]));
    }
  }
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "expand_as", expand_as);
    *out =
        expand(dev_ctx,
               x,
               std::vector<int64_t>(target_shape.begin(), target_shape.end()));
    PADDLE_GCU_KERNEL_END("expand_as", expand_as);
  } else {
    phi::DDim out_dims = phi::make_ddim(target_shape);

    out->Resize(out_dims);

    dev_ctx.template Alloc<T>(out);

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["target_shape"] = target_shape;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "expand_as_v2",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(expand_as,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ExpandAsKernel,
                          int8_t,
                          uint8_t,
                          int,
                          float,
                          phi::dtype::float16) {}
