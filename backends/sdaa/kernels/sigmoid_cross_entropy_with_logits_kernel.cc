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

const int kIgnoreIndex = -100;

void CheckAttrs(bool normalize, int ignore_index) {
  // Add this check is is due to sdaa SigmoidCrossEntropyWithLogits
  // and SigmoidCrossEntropyWithLogitsGrad does't supoort
  // attr normalize and ignore_index
  PADDLE_ENFORCE_EQ(normalize,
                    false,
                    phi::errors::InvalidArgument(
                        "attr normalize must be false, but got true"));
  PADDLE_ENFORCE_EQ(ignore_index,
                    kIgnoreIndex,
                    phi::errors::InvalidArgument(
                        "attr ignore_index must be default %d, but got %d",
                        kIgnoreIndex,
                        ignore_index));
}

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const phi::DenseTensor& label,
    const paddle::optional<phi::DenseTensor>& pos_weight,
    bool normalize,
    int ignore_index,
    phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SigmoidCrossEntropyWithLogitsKernel";
  CheckAttrs(normalize, ignore_index);

  dev_ctx.template Alloc<T>(out);
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  int n = 1, x_size = x.dims().size();
  for (int i = 0; i < x_size - 1; ++i) {
    n *= x.dims()[i];
  }
  int d = x.dims()[x_size - 1];
  phi::DenseTensor w;
  std::vector<int> w_dims = {n, d};
  phi::DDim w_dim = phi::make_ddim(w_dims);
  phi::DenseTensorMeta w_meta = {x.dtype(), w_dim};
  w.set_meta(w_meta);
  dev_ctx.template Alloc<T>(&w);
  sdaa_ops::doFillTensor<T>(dev_ctx, static_cast<T>(1), x.dtype(), &w);
  phi::DenseTensor p_w;
  w_dims = {d};
  w_dim = phi::make_ddim(w_dims);
  w_meta = {x.dtype(), w_dim};
  if (pos_weight) {
    p_w = pos_weight.get();
  } else {
    p_w.set_meta(w_meta);
    dev_ctx.template Alloc<T>(&p_w);
    sdaa_ops::doFillTensor<T>(dev_ctx, static_cast<T>(1), x.dtype(), &p_w);
  }
  tecodnnTensorDescriptor_t x_Desc;
  tecodnnTensorDescriptor_t w_Desc;
  TECODNN_CHECK(tecodnnCreateTensorDescriptor(&x_Desc));
  TECODNN_CHECK(tecodnnCreateTensorDescriptor(&w_Desc));
  int x_shape[2] = {n, d};
  int w_shape[1] = {d};
  tecodnnLossReductionMode_t reduction = TECODNN_LOSS_REDUCTION_NONE;
  tecodnnDataType_t dt =
      sdaa_ops::ToTecodnnDataType(phi::CppTypeToDataType<T>::Type());
  TECODNN_CHECK(tecodnnSetTensorNdDescriptorEx(
      x_Desc, TECODNN_TENSOR_NHWC, dt, 2, x_shape));
  TECODNN_CHECK(tecodnnSetTensorNdDescriptorEx(
      w_Desc, TECODNN_TENSOR_NHWC, dt, 1, w_shape));
  TECODNN_CHECK(tecodnnSigmoidBCELossForward(tecodnnHandle,
                                             reduction,
                                             x_Desc,
                                             x.data(),
                                             x_Desc,
                                             label.data(),
                                             x_Desc,
                                             w.data(),
                                             w_Desc,
                                             p_w.data(),
                                             x_Desc,
                                             out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(w_Desc));
}

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const phi::DenseTensor& label,
    const paddle::optional<phi::DenseTensor>& pos_weight,
    const phi::DenseTensor& dout,
    bool normalize,
    int ignore_index,
    phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA SigmoidCrossEntropyWithLogitsGradKernel";
  CheckAttrs(normalize, ignore_index);

  dev_ctx.template Alloc<T>(dx);
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  int n = 1, x_size = x.dims().size();
  for (int i = 0; i < x_size - 1; ++i) {
    n *= x.dims()[i];
  }
  int d = x.dims()[x_size - 1];
  phi::DenseTensor w;
  std::vector<int> w_dims = {n, d};
  phi::DDim w_dim = phi::make_ddim(w_dims);
  phi::DenseTensorMeta w_meta = {x.dtype(), w_dim};
  w.set_meta(w_meta);
  dev_ctx.template Alloc<T>(&w);
  sdaa_ops::doFillTensor<T>(dev_ctx, static_cast<T>(1), x.dtype(), &w);
  phi::DenseTensor p_w;
  w_dims = {d};
  w_dim = phi::make_ddim(w_dims);
  w_meta = {x.dtype(), w_dim};
  if (pos_weight) {
    p_w = pos_weight.get();
  } else {
    p_w.set_meta(w_meta);
    dev_ctx.template Alloc<T>(&p_w);
    sdaa_ops::doFillTensor<T>(dev_ctx, static_cast<T>(1), x.dtype(), &p_w);
  }
  tecodnnTensorDescriptor_t x_Desc;
  tecodnnTensorDescriptor_t w_Desc;
  TECODNN_CHECK(tecodnnCreateTensorDescriptor(&x_Desc));
  TECODNN_CHECK(tecodnnCreateTensorDescriptor(&w_Desc));
  int x_shape[2] = {n, d};
  int w_shape[1] = {d};
  tecodnnLossReductionMode_t reduction = TECODNN_LOSS_REDUCTION_NONE;
  tecodnnDataType_t dt =
      sdaa_ops::ToTecodnnDataType(phi::CppTypeToDataType<T>::Type());
  TECODNN_CHECK(tecodnnSetTensorNdDescriptorEx(
      x_Desc, TECODNN_TENSOR_NHWC, dt, 2, x_shape));
  TECODNN_CHECK(tecodnnSetTensorNdDescriptorEx(
      w_Desc, TECODNN_TENSOR_NHWC, dt, 1, w_shape));
  TECODNN_CHECK(tecodnnSigmoidBCELossBackward(tecodnnHandle,
                                              reduction,
                                              x_Desc,
                                              x.data(),
                                              x_Desc,
                                              label.data(),
                                              x_Desc,
                                              w.data(),
                                              w_Desc,
                                              p_w.data(),
                                              x_Desc,
                                              dout.data(),
                                              x_Desc,
                                              dx->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(w_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sigmoid_cross_entropy_with_logits,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidCrossEntropyWithLogitsKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(
    sigmoid_cross_entropy_with_logits_grad,
    sdaa,
    ALL_LAYOUT,
    custom_kernel::SigmoidCrossEntropyWithLogitsGradKernel,
    float,
    phi::dtype::float16) {}
