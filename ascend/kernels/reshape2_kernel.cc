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

#include "npu_funcs.h"
#include "npu_op_runner.h"

#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/meta_tensor.h"

namespace custom_kernel {
static phi::DDim ValidateShape(const std::vector<int64_t> shape,
                               const phi::DDim& in_dims) {
  const int64_t in_size = phi::product(in_dims);
  auto in_dims_vec = phi::vectorize(in_dims);
  bool all_positive = std::all_of(in_dims_vec.cbegin(),
                                  in_dims_vec.cend(),
                                  [](int64_t i) { return i > 0; });
  // only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int64_t unk_dim_val = -1;
  const int64_t copy_dim_val = 0;

  std::vector<int64_t> output_shape(shape.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      PADDLE_ENFORCE_EQ(
          unk_dim_idx,
          -1,
          phi::errors::InvalidArgument(
              "Only one dimension value of 'shape' in ReshapeOp can "
              "be -1. But received shape = [%s], shape[%d] is also -1.",
              phi::make_ddim(shape),
              i));
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      PADDLE_ENFORCE_LT(
          static_cast<int>(i),
          in_dims.size(),
          phi::errors::InvalidArgument(
              "The index of 0 in `shape` must be less than "
              "the input tensor X's dimensions. "
              "But received shape = [%s], shape[%d] = 0, X's shape = [%s], "
              "X's dimensions = %d.",
              phi::make_ddim(shape),
              i,
              in_dims,
              in_dims.size()));
    } else {
      PADDLE_ENFORCE_GT(
          shape[i],
          0,
          phi::errors::InvalidArgument(
              "Each dimension value of 'shape' in ReshapeOp must not "
              "be negative except one unknown dimension. "
              "But received  shape = [%s], shape[%d] = %d.",
              phi::make_ddim(shape),
              i,
              shape[i]));
    }

    // NOTE all non-zero values will be converted to True (include negative
    // value)
    capacity *= (shape[i] ? shape[i] : in_dims[i]);
    output_shape[i] = (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);
  }

  if (unk_dim_idx != -1) {
    if (all_positive) {
      // in_size < 0 and is un-determinate in compile time, skip the check,
      // for example, in_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, in_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_shape[unk_dim_idx] = -in_size / capacity;
      PADDLE_ENFORCE_EQ(
          output_shape[unk_dim_idx] * capacity,
          -in_size,
          phi::errors::InvalidArgument(
              "The 'shape' attribute in ReshapeOp is invalid. "
              "The input tensor X'size must be divisible by known "
              "capacity of 'shape'. "
              "But received X's shape = [%s], X's size = %d, "
              "'shape' is [%s], known capacity of 'shape' is %d.",
              in_dims,
              in_size,
              phi::make_ddim(shape),
              capacity));
    } else {
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    if (all_positive) {
      PADDLE_ENFORCE_EQ(
          capacity,
          in_size,
          phi::errors::InvalidArgument(
              "The 'shape' in ReshapeOp is invalid. "
              "The input tensor X'size must be equal to the capacity of "
              "'shape'. "
              "But received X's shape = [%s], X's size = %d, 'shape' is "
              "[%s], the capacity of 'shape' is %d.",
              in_dims,
              in_size,
              phi::make_ddim(shape),
              capacity));
    }
  }

  // support reshape with zero-input(input tensor with product(shape) == 0)
  // by now we require that if the input tensor is zero shape, the target
  // shape of output must be zero
  if (in_size == 0) {
    PADDLE_ENFORCE_LE(
        capacity,
        in_size,
        phi::errors::InvalidArgument(
            "The 'shape' in ReshapeOp is invalid. "
            "The input tensor X's shape = [%s], X's capacity = %d."
            "But the target shape of Out is [%s],  the "
            "capacity of 'Out' is %d.",
            in_dims,
            in_size,
            phi::make_ddim(shape),
            capacity));
  }

  return phi::make_ddim(output_shape);
}

void InferMetaFromVecValue(const phi::MetaTensor& x,
                           const std::vector<int64_t>& shape,
                           phi::MetaTensor* out) {
  PADDLE_ENFORCE_EQ(!shape.empty(),
                    true,
                    phi::errors::InvalidArgument(
                        "The parameter 'shape' in ReshapeOp must be set. "
                        "But received 'shape' is empty."));
  auto x_dims = x.dims();
  auto out_dims = ValidateShape(shape, x_dims);
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  if (x_dims[0] == out_dims[0]) {
    // Only pass LoD when the first dimension of output and Input(X)
    // are the same.
    out->share_lod(x);
  }
}

template <typename T, typename Context>
void ReshapeWithXShapeKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::ScalarArray& shape,
                             phi::DenseTensor* xshape,
                             phi::DenseTensor* out) {
  phi::MetaTensor meta_out(out);
  custom_kernel::InferMetaFromVecValue(x, shape.GetData(), &meta_out);
  if (x.initialized() && x.Holder() == out->Holder()) {
    dev_ctx.Alloc(out, x.dtype());
    return;
  }
  dev_ctx.Alloc(out, x.dtype());

  auto dims = out->dims();
  TensorCopy(dev_ctx, x, false, out);
  out->Resize(dims);
  // out->ResetLoD(x.lod());
}

template <typename T, typename Context>
void ReshapeGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out_grad,
                       phi::DenseTensor* x_grad) {
  dev_ctx.Alloc(x_grad, out_grad.dtype());
  auto x_dims = x_grad->dims();
  TensorCopy(dev_ctx, out_grad, false, x_grad);
  x_grad->Resize(x_dims);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(reshape_with_xshape,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::ReshapeWithXShapeKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}
PD_REGISTER_PLUGIN_KERNEL(reshape_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::ReshapeGradKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}
