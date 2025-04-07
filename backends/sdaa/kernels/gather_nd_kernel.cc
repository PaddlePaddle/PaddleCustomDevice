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

namespace custom_kernel {

template <typename T, typename Context>
void GatherNdKernel(const Context &dev_ctx,
                    const phi::DenseTensor &x,
                    const phi::DenseTensor &index,
                    phi::DenseTensor *out) {
  VLOG(4) << "Call SDAA GatherNdKernel";

  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 0) return;
  if (index.numel() == 0) {
    phi::DenseTensor x_temp(x);
    sdaa_ops::doExpandTensor(dev_ctx, x_temp, out);
    return;
  }

  const auto &index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      phi::errors::InvalidArgument("Index holds the wrong type, it holds [%s],"
                                   "but desires to be [%s] or [%s]",
                                   index_type,
                                   phi::DataType::INT32,
                                   phi::DataType::INT64));

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> index_dims = phi::vectorize<int>(index.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());
  if (out_dims.empty()) {  // support scalar output
    out_dims.push_back(1);
  }
  tecodnnTensorDescriptor_t xDesc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t indexDesc = sdaa_ops::GetTecodnnTensorDesc(
      index_dims, index.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t outDesc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::Undefined);
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  TECODNN_CHECK(tecodnnGatherNdForward(tecodnnHandle,
                                       xDesc,
                                       x.data(),
                                       indexDesc,
                                       index.data(),
                                       outDesc,
                                       out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(xDesc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(indexDesc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(outDesc));
}

template <typename T, typename Context>
void GatherNdGradKernel(const Context &ctx,
                        const phi::DenseTensor &x UNUSED,
                        const phi::DenseTensor &index,
                        const phi::DenseTensor &out_grad,
                        phi::DenseTensor *x_grad) {
  VLOG(4) << "Call SDAA GatherNdGradKernel";

  ctx.template Alloc<T>(x_grad);

  if (out_grad.numel() == 0) return;

  if (index.numel() == 0) {
    int diff = out_grad.dims().size() - x_grad->dims().size();
    if (diff == 0) {
      TensorCopy(ctx, out_grad, false, x_grad);
    } else {
      std::vector<int64_t> axes;
      for (size_t i = 0; i < diff; ++i) {
        axes.push_back(i);
      }
      sdaa_ops::doSumTensor(ctx, out_grad, axes, x_grad);
    }
    return;
  }

  auto index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      phi::errors::InvalidArgument("Index holds the wrong type, it holds [%s],"
                                   "but desires to be [%s] or [%s]",
                                   index_type,
                                   phi::DataType::INT32,
                                   phi::DataType::INT64));

  std::vector<int> index_dims = phi::vectorize<int>(index.dims());
  std::vector<int> out_grad_dims = phi::vectorize<int>(out_grad.dims());
  tecodnnTensorDescriptor_t index_desc = sdaa_ops::GetTecodnnTensorDesc(
      index_dims, index.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_grad_desc = sdaa_ops::GetTecodnnTensorDesc(
      out_grad_dims, out_grad.dtype(), TensorFormat::Undefined);

  std::vector<int> x_grad_dims = phi::vectorize<int>(x_grad->dims());
  tecodnnTensorDescriptor_t x_grad_desc = sdaa_ops::GetTecodnnTensorDesc(
      x_grad_dims, x_grad->dtype(), TensorFormat::Undefined);

  phi::DenseTensor x_tmp;
  phi::DenseTensorMeta temp_x_meta = {x_grad->dtype(), x_grad->dims()};
  x_tmp.set_meta(temp_x_meta);
  ctx.template Alloc<T>(&x_tmp);
  sdaa_ops::doFillTensor<T>(ctx, static_cast<T>(0), x_grad->dtype(), &x_tmp);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(ctx);
  TECODNN_CHECK(tecodnnScatterNdAdd(tecodnnHandle,
                                    x_grad_desc,
                                    x_tmp.data(),
                                    index_desc,
                                    index.data(),
                                    out_grad_desc,
                                    out_grad.data(),
                                    x_grad_desc,
                                    x_grad->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(index_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_grad_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_grad_desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gather_nd,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GatherNdKernel,
                          float,
                          phi::dtype::float16,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(gather_nd_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GatherNdGradKernel,
                          float,
                          phi::dtype::float16,
                          int,
                          int64_t) {}
