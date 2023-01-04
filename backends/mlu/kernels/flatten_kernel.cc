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

#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

static std::vector<int32_t> GetOutputShape(const int start_axis,
                                           const int stop_axis,
                                           const phi::DDim& in_dims) {
  if (in_dims.size() == 0) {
    return {1};
  }

  int64_t outer = 1;
  std::vector<int32_t> out_shape;
  int in_dims_size = in_dims.size();
  out_shape.reserve(in_dims_size - stop_axis + start_axis);
  int real_start_axis = start_axis, real_stop_axis = stop_axis;
  if (start_axis < 0) {
    real_start_axis = start_axis + in_dims_size;
  }
  if (stop_axis < 0) {
    real_stop_axis = stop_axis + in_dims_size;
  }

  for (int i = 0; i < real_start_axis; ++i) {
    out_shape.push_back(in_dims[i]);
  }
  for (int i = real_start_axis; i <= real_stop_axis; i++) {
    if (in_dims[i] == -1 || outer == -1) {
      outer = -1;
    } else {
      outer *= in_dims[i];
    }
  }
  out_shape.push_back(outer);
  for (int i = real_stop_axis + 1; i < in_dims_size; i++) {
    out_shape.push_back(in_dims[i]);
  }

  return out_shape;
}

inline void SetXShape(const phi::DenseTensor& x, phi::DenseTensor* xshape) {
  const auto& in_dims = x.meta().dims;
  std::vector<int64_t> xshape_dims(in_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    xshape_dims[i + 1] = in_dims[i];
  }
  xshape->ResizeAndAllocate(phi::make_ddim(xshape_dims));
  xshape->ResetLoD(x.meta().lod);
}

template <typename T, typename Context>
void FlattenKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int start_axis,
                   int stop_axis,
                   phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  // make out dims
  auto in_dims = x.dims();
  auto out_dims =
      phi::make_ddim(GetOutputShape(start_axis, stop_axis, in_dims));
  TensorCopy(dev_ctx, x, false, out);
  out->Resize(out_dims);
}

template <typename T, typename Context>
void FlattenGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& xshape,
                       const phi::DenseTensor& out_grad,
                       phi::DenseTensor* x_grad) {
  auto xshape_dims = xshape.dims();
  auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());

  TensorCopy(dev_ctx, out_grad, false, x_grad);
  x_grad->Resize(x_dims);
}

template <typename T, typename Context>
void FlattenWithXShape(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       int start_axis,
                       int stop_axis,
                       phi::DenseTensor* out,
                       phi::DenseTensor* xshape) {
  custom_kernel::FlattenKernel<T, Context>(
      dev_ctx, x, start_axis, stop_axis, out);
  SetXShape(x, xshape);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(flatten,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::FlattenKernel,
                          float,
                          double,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(flatten_with_xshape,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::FlattenWithXShape,
                          float,
                          double,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(flatten_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::FlattenGradKernel,
                          float,
                          double,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::float16) {}
