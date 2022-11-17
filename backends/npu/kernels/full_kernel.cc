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
#include "kernels/funcs/op_command.h"
#include "paddle/phi/core/tensor_meta.h"

namespace custom_kernel {

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& val,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  T value = val.to<T>();

  using CommonType = typename std::common_type<
      float,
      typename std::conditional<std::is_same<T, phi::dtype::float16>::value,
                                float,
                                T>::type>::type;

  auto common_type_value = static_cast<CommonType>(value);

  // Check whether the filled value is valid
  bool is_out_range = true;
  if (std::isinf(value) || std::isnan(value)) {
    is_out_range = false;
  }

  if ((common_type_value >=
       static_cast<CommonType>(std::numeric_limits<T>::lowest())) &&
      (common_type_value <=
       static_cast<CommonType>(std::numeric_limits<T>::max()))) {
    is_out_range = false;
  }

  PADDLE_ENFORCE_EQ(
      is_out_range,
      false,
      phi::errors::InvalidArgument(
          "The filled value is out of range for target type, "
          "current kernel type is %s, the range should between %f "
          "and %f, but now value is %f.",
          typeid(T).name(),
          static_cast<CommonType>(std::numeric_limits<T>::lowest()),
          static_cast<CommonType>(std::numeric_limits<T>::max()),
          static_cast<CommonType>(value)));

  PADDLE_ENFORCE_EQ(std::isnan(value),
                    false,
                    phi::errors::InvalidArgument("The filled value is NaN."));
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor value_tensor;
  value_tensor.Resize({1});
  dev_ctx.template HostAlloc<T>(&value_tensor);
  *(value_tensor.data<T>()) = val.to<T>();

  if (out->numel() == 1) {
    GRAPH_RUN({
      experimental::OpCommand("Const")
          .Output(*out,
                  experimental::TensorDescMaker("y", *out).SetDataLayout(
                      phi::DataLayout::ANY))
          .Attr("value", value_tensor)
          .Run(dev_ctx);
    });
    ACL_RUN({ TensorCopy(dev_ctx, value_tensor, false, out); });
  } else {
    // NOTE(wangran16): There is a bug when fill a tensor of dim [1]
    phi::DenseTensor x_dims;
    TensorFromVector(
        dev_ctx, phi::vectorize(x.dims()), phi::CPUContext(), &x_dims);

    experimental::OpCommand("Fill")
        .Input(x_dims,
               experimental::TensorDescMaker("dims", x_dims)
                   .SetDataLayout(phi::DataLayout::ANY))
        .ScalarInput(value_tensor,
                     experimental::TensorDescMaker("value", value_tensor)
                         .SetDataLayout(phi::DataLayout::ANY))
        .Output(*out,
                experimental::TensorDescMaker("y", *out).SetDataLayout(
                    phi::DataLayout::ANY))
        .Run(dev_ctx);
  }
}

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  phi::DenseTensor x_dims;
  x_dims.Resize(phi::make_ddim(shape.GetData()));

  custom_kernel::FullLikeKernel<T, Context>(dev_ctx, x_dims, val, dtype, out);
}

template <typename T, typename Context>
void FullBatchSizeLikeKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const std::vector<int>& shape,
                             const phi::Scalar& val,
                             phi::DataType dtype,
                             int x_batch_size_dim,
                             int out_batch_size_dim,
                             phi::DenseTensor* out) {
  if (x.lod().size() && x_batch_size_dim == 0) {
    // set the correct batch size for the LoDTensor.
    auto odims = out->dims();
    odims[out_batch_size_dim] = static_cast<int>(x.lod().back().size()) - 1;
    custom_kernel::FullKernel<T, Context>(
        dev_ctx, phi::vectorize(odims), val, dtype, out);
  } else {
    auto odims = out->dims();
    odims[out_batch_size_dim] = x.dims()[x_batch_size_dim];
    custom_kernel::FullKernel<T, Context>(
        dev_ctx, phi::vectorize(odims), val, dtype, out);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(full,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FullKernel,
                          bool,
                          int16_t,
                          int32_t,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(full_like,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FullLikeKernel,
                          bool,
                          int16_t,
                          int32_t,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(full_batch_size_like,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FullBatchSizeLikeKernel,
                          bool,
                          int16_t,
                          int32_t,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
