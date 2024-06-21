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

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
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
  VLOG(4) << "Call SDAA TopkKernel";
  int xDims = x.dims().size();
  if (axis < 0) {
    axis += xDims;
  }
  PADDLE_ENFORCE_EQ(
      xDims <= 4,
      true,
      phi::errors::InvalidArgument(
          "input dimension must be less than or equal to 4 on SDAA place"));

  int k = k_scalar.to<int>();

  phi::DDim output_dims = x.dims();
  output_dims[axis] = k;
  out->Resize(output_dims);
  indices->Resize(output_dims);

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<int64_t>(indices);

  // support 0D tensor
  if (output_dims.size() == 0) {
    phi::Copy(dev_ctx, x, out->place(), false, out);
    sdaa_ops::doFillTensor<int64_t>(
        dev_ctx, static_cast<int64_t>(0.0), indices->dtype(), indices);
    out->Resize(output_dims);
    indices->Resize(output_dims);
    return;
  }

  // custom topk op only supports one dim, float32 and largest == True
  if (1 == xDims && phi::DataType::FLOAT32 == x.dtype() && largest) {
    std::vector<int> x_dimensions = phi::vectorize<int>(x.dims());
    std::vector<int> y_dimensions = phi::vectorize<int>(output_dims);
    std::vector<int> indices_dimensions = phi::vectorize<int>(output_dims);

    phi::DenseTensor* in = const_cast<phi::DenseTensor*>(&x);

    tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

    tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
        x_dimensions, x.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t y_Desc = sdaa_ops::GetTecodnnTensorDesc(
        y_dimensions, out->dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t indices_Desc = sdaa_ops::GetTecodnnTensorDesc(
        y_dimensions, phi::DataType::INT64, TensorFormat::Undefined);

    // topk description
    tecodnnTopkDescriptor_t topk_Desc;
    TECODNN_CHECK(tecodnnCreateTopkDescriptor(&topk_Desc));

    tecodnnTopkSortMode_t sort_mode =
        largest ? tecodnnTopkSortMode_t::TECODNN_TOPK_SORT_LARGEST
                : tecodnnTopkSortMode_t::TECODNN_TOPK_SORT_SMALLEST;
    tecodnnTopkOrderMode_t order_mode =
        sorted ? tecodnnTopkOrderMode_t::TECODNN_TOPK_RESULT_ORDERED
               : tecodnnTopkOrderMode_t::TECODNN_TOPK_RESULT_DISORDERED;
    TECODNN_CHECK(
        tecodnnSetTopkDescriptor(topk_Desc, axis, k, sort_mode, order_mode));

    // workspace size
    size_t workspace_size;
    tecodnnGetTopkWorkspaceSize(
        topk_Desc, x_Desc, y_Desc, indices_Desc, &workspace_size);

    phi::DenseTensor dev_workspace;
    dev_workspace.Resize(
        phi::make_ddim({static_cast<int64_t>(workspace_size)}));
    dev_ctx.Alloc(&dev_workspace, phi::DataType::INT8);

    tecodnnCustomTopk(tecodnnHandle,
                      topk_Desc,
                      x_Desc,
                      in->data(),
                      y_Desc,
                      out->data(),
                      indices_Desc,
                      indices->data(),
                      dev_workspace.data(),
                      workspace_size);

    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(indices_Desc));
    TECODNN_CHECK(tecodnnDestroyTopkDescriptor(topk_Desc));
    return;
  }

  // the indices param in the tecodnnTopk is int.
  phi::DenseTensor indices_int;
  indices_int.Resize(indices->dims());
  dev_ctx.template Alloc<int32_t>(&indices_int);

  std::vector<int> x_dimensions = phi::vectorize<int>(x.dims());
  std::vector<int> y_dimensions = phi::vectorize<int>(output_dims);

  phi::DenseTensor* in = const_cast<phi::DenseTensor*>(&x);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dimensions, x.dtype(), TensorFormat::NCHW);
  tecodnnTensorDescriptor_t y_Desc = sdaa_ops::GetTecodnnTensorDesc(
      y_dimensions, out->dtype(), TensorFormat::NCHW);

  axis += 4 - xDims;
  TECODNN_CHECK(tecodnnTopk(tecodnnHandle,
                            axis,
                            k,
                            static_cast<int>(largest),
                            static_cast<int>(sorted),
                            x_Desc,
                            in->data(),
                            y_Desc,
                            out->data(),
                            indices_int.data()));

  sdaa_ops::doCastTensor(dev_ctx, indices_int, indices);

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
}

template <typename T, typename Context>
void TopkGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& indices,
                    const phi::DenseTensor& out_grad,
                    const phi::Scalar& k_scalar,
                    int axis,
                    bool largest UNUSED,
                    bool sorted UNUSED,
                    phi::DenseTensor* x_grad) {
  VLOG(4) << "Call SDAA TopkGradKernel";
  const auto& in_dim_size = x.dims().size();
  // axis < 0, get the real axis
  axis = (axis < 0) ? (in_dim_size + axis) : axis;

  PADDLE_ENFORCE_EQ(
      in_dim_size <= 4,
      true,
      phi::errors::InvalidArgument(
          "input dimension must be less than or equal to 4 on SDAA place"));

  int k = k_scalar.to<int>();

  phi::DDim x_grad_dims = x.dims();
  x_grad->Resize(x_grad_dims);

  dev_ctx.template Alloc<T>(x_grad);

  // if not memset 0, it will generate wrong x_grad in non-blocking launch
  sdaa_ops::doMemsetTensor(dev_ctx, static_cast<int>(0), x_grad);

  // support 0D tensor
  if (x_grad_dims.size() == 0) {
    phi::Copy(dev_ctx, x, x_grad->place(), false, x_grad);
    return;
  }

  std::vector<int> x_dimensions = phi::vectorize<int>(x.dims());
  std::vector<int> indices_dimensions = phi::vectorize<int>(indices.dims());
  std::vector<int> out_grad_dimensions = phi::vectorize<int>(out_grad.dims());
  std::vector<int> x_grad_dimensions = phi::vectorize<int>(x_grad->dims());

  phi::DenseTensor* in = const_cast<phi::DenseTensor*>(&x);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t x_desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dimensions, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t indices_desc = sdaa_ops::GetTecodnnTensorDesc(
      indices_dimensions, indices.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_grad_desc = sdaa_ops::GetTecodnnTensorDesc(
      out_grad_dimensions, out_grad.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t x_grad_desc = sdaa_ops::GetTecodnnTensorDesc(
      x_grad_dimensions, x_grad->dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnCustomTopkGrad(tecodnnHandle,
                                      x_desc,
                                      x.data(),
                                      indices_desc,
                                      indices.data(),
                                      out_grad_desc,
                                      out_grad.data(),
                                      k,
                                      axis,
                                      x_grad_desc,
                                      x_grad->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(indices_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_grad_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_grad_desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(topk,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::TopkKernel,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}

PD_REGISTER_PLUGIN_KERNEL(topk_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::TopkGradKernel,
                          float,
                          phi::dtype::float16,
                          int,
                          int64_t) {}
