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

#include "kernels/funcs/mlu_baseop.h"

namespace custom_kernel {

template <typename T, typename Context>
void ExpandKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& shape,
                  phi::DenseTensor* out) {
  auto in_dims = x.dims();
  auto expand_shape = shape.GetData();
  auto vec_in_dims = phi::vectorize<int>(in_dims);
  auto diff = expand_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  std::vector<int> final_expand_shape(vec_in_dims.size());
  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    PADDLE_ENFORCE_NE(
        expand_shape[i],
        0,
        phi::errors::InvalidArgument("The expanded size cannot be zero."));
    if (i < diff) {  // expand_shape = [3,4,-1,-1], x = [10,2] -->
                     // final_expand_shape = [3,4,10,2]
      PADDLE_ENFORCE_GT(
          expand_shape[i],
          0,
          phi::errors::InvalidArgument(
              "The expanded size (%d) for non-existing dimensions must be "
              "positive for expand_v2 op.",
              expand_shape[i]));
      final_expand_shape[i] = expand_shape[i];
    } else if (expand_shape[i] > 0) {  // expand_shape = [3,4,10,4], x =
                                       // [10,1] --> final_expand_shape =
                                       // [3,4,10,4]
      if (vec_in_dims[i] != 1) {
        PADDLE_ENFORCE_EQ(
            vec_in_dims[i],
            expand_shape[i],
            phi::errors::InvalidArgument(
                "The value (%d) of the non-singleton dimension does not match"
                " the corresponding value (%d) in shape for expand_v2 op.",
                vec_in_dims[i],
                expand_shape[i]));
        final_expand_shape[i] = expand_shape[i];
      } else {
        final_expand_shape[i] = expand_shape[i];
      }
    } else {  // expand_shape = [3,4,-1,-1], x = [10,2] --> final_expand_shape
              // = [3,4,10,2]
      PADDLE_ENFORCE_EQ(
          expand_shape[i],
          -1,
          phi::errors::InvalidArgument(
              "When the value in shape is negative for expand_v2 op, "
              "only -1 is supported, but the value received is %d.",
              expand_shape[i]));
      final_expand_shape[i] = vec_in_dims[i];
    }
  }

  auto rank = x.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      0,
      phi::errors::InvalidArgument(
          "The rank of the input 'x' for expand_v2_mlu op must be positive, "
          "but the value received is %d.",
          rank));
  auto shape_size = final_expand_shape.size();
  PADDLE_ENFORCE_GE(
      shape_size,
      rank,
      phi::errors::InvalidArgument(
          "The number (%d) of elements of 'shape' for expand_v2_mlu op must "
          "be "
          "greater than or equal to the rank (%d) of the input 'x'.",
          shape_size,
          rank));

  phi::DDim out_dims = phi::make_ddim(final_expand_shape);
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::BroadcastTo(
      dev_ctx, x_desc.get(), GetBasePtr(&x), out_desc.get(), GetBasePtr(out));
}

// template <typename T, typename Context>
// void ExpandGradKernel(const Context& dev_ctx,
//                       const phi::DenseTensor& x,
//                       const phi::DenseTensor& out_grad,
//                       const phi::IntArray& shape,
//                       phi::DenseTensor* in_grad) {
// }

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(expand,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ExpandKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(expand_grad,
//                           CustomMLU,
//                           ALL_LAYOUT,
//                           custom_kernel::ExpandGradKernel,
//                           int,
//                           float,
//                           phi::dtype::float16) {}
