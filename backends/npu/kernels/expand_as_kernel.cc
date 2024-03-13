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

#define MAX_RANK_SUPPORTED 6

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
bool check_tensor_values_in_range(const Context& dev_ctx,
                                  const phi::DenseTensor& x,
                                  phi::DataType dtype = phi::DataType::INT32) {
  if (x.dtype() != phi::DataType::INT64) {
    return true;
  }
  std::vector<int64_t> x_v;
  TensorToVector(dev_ctx, x, dev_ctx, &x_v);
  if (static_cast<int32_t>(x_v[0]) != x_v[0]) {
    return false;
  }
  return true;
}

template <typename T, typename Context>
void ExpandAsKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const paddle::optional<phi::DenseTensor>& y,
                    const std::vector<int>& target_shape,
                    phi::DenseTensor* out) {
  bool x_inrange = check_tensor_values_in_range<T, Context>(dev_ctx, x);
  PADDLE_ENFORCE_EQ(
      x_inrange,
      1,
      phi::errors::InvalidArgument("The size of the input int64 data must be "
                                   "whithin the range of int32."));

  phi::DenseTensor cast_x;

  if (x.dtype() == phi::DataType::INT64) {
    phi::DenseTensorMeta meta(x.meta());
    meta.dtype = phi::DataType::INT32;
    cast_x.set_meta(meta);

    custom_kernel::CastKernel<T, Context>(
        dev_ctx, x, phi::DataType::INT32, &cast_x);
  } else {
    cast_x = x;
  }
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

  phi::DDim out_dims = phi::make_ddim(target_shape);

  out->Resize(out_dims);

  dev_ctx.template Alloc<T>(out);

  const auto& runner =
      NpuOpRunner("ExpandD", {cast_x}, {*out}, {{"shape", target_shape}});

  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(expand_as,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ExpandAsKernel,
                          int8_t,
                          uint8_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
