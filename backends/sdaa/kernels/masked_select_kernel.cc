// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

void doMaskedSelectOpTensor(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& mask,
                            phi::DenseTensor* out) {
  VLOG(4) << "tecodnn selectmask op called";

  auto x_dim = x.dims();
  auto mask_dim = mask.dims();

  std::vector<int> x_dims = phi::vectorize<int>(x_dim);
  std::vector<int> mask_dims = phi::vectorize<int>(mask_dim);
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  phi::DenseTensor mask_int;
  mask_int.Resize(mask_dim);
  dev_ctx.Alloc(&mask_int, DataType::UINT8);
  sdaa_ops::doCastTensor(dev_ctx, mask, &mask_int);

  tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc =
      sdaa_ops::GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t mask_Desc = sdaa_ops::GetTecodnnTensorDesc(
      mask_dims, mask_int.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::NHWC);
  // int selectCount = 0;
  phi::DenseTensor selectCount;
  selectCount.Resize(phi::make_ddim({1}));
  dev_ctx.Alloc(&selectCount, DataType::INT32);

  TECODNN_CHECK(tecodnnMaskedSelect(handle,
                                    x_Desc,
                                    x.data(),
                                    mask_Desc,
                                    mask_int.data(),
                                    out_Desc,
                                    out->data(),
                                    selectCount.data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(mask_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

template <typename T, typename Context>
void MaskedSelectKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& mask,
                        phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA MaskedSelectKernel";
  auto input_dim = x.dims();
  auto mask_dim = mask.dims();
  PADDLE_ENFORCE_EQ(input_dim,
                    mask_dim,
                    phi::errors::InvalidArgument(
                        "The dim size of input and mask in OP(masked_selected) "
                        "must be equal, but got input dim:(%ld), mask dim: "
                        "(%ld). Please check input "
                        "value.",
                        input_dim,
                        mask_dim));

  phi::DenseTensor mask_int;
  mask_int.Resize(mask_dim);
  dev_ctx.Alloc(&mask_int, DataType::INT64);
  sdaa_ops::doCastTensor(dev_ctx, mask, &mask_int);

  phi::DenseTensor nonzconunt;
  nonzconunt.Resize(phi::make_ddim({1}));
  dev_ctx.Alloc(&nonzconunt, DataType::INT64);

  std::vector<int64_t> reduce_dims;
  int ndims = mask_dim.size();
  for (size_t i = 0; i < ndims; i++) {
    reduce_dims.push_back(i);
  }

  sdaa_ops::doSumTensor(dev_ctx, mask_int, reduce_dims, &nonzconunt);
  std::vector<int64_t> out_size;
  custom_kernel::TensorToVector(dev_ctx, nonzconunt, dev_ctx, &out_size);
  // VLOG(4) << "out_size" << out_size[0];
  out->Resize(phi::make_ddim({out_size[0]}));
  dev_ctx.template Alloc<T>(out);
  custom_kernel::doMaskedSelectOpTensor(dev_ctx, x, mask, out);
}

template <typename T, typename Context>
void MaskedSelectGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& mask,
                            const phi::DenseTensor& out_grad,
                            phi::DenseTensor* x_grad) {
  VLOG(4) << "CALL SDAA MaskedSelectGradKernel";

  auto mask_size = mask.numel();
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> mask_dims = phi::vectorize<int>(mask.dims());
  std::vector<int> out_grad_dims = phi::vectorize<int>(out_grad.dims());
  std::vector<int> x_grad_dims = phi::vectorize<int>(x_grad->dims());
  dev_ctx.template Alloc<T>(x_grad);
  if (mask_size <= 0) return;

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t mask_Desc =
      sdaa_ops::GetTecodnnBoolTensorDesc(mask_dims, TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_grad_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_grad_dims, out_grad.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t x_grad_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_grad_dims, x_grad->dtype(), TensorFormat::Undefined);
  TECODNN_CHECK(tecodnnMaskedSelectBackward(tecodnnHandle,
                                            x_Desc,
                                            nullptr,
                                            mask_Desc,
                                            mask.data(),
                                            out_grad_Desc,
                                            out_grad.data(),
                                            x_grad_Desc,
                                            x_grad->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(mask_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_grad_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_grad_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(masked_select,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MaskedSelectKernel,
                          float,
                          phi::dtype::float16,
                          uint8_t) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(masked_select_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MaskedSelectGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
