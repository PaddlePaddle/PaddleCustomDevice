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

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
namespace custom_kernel {

template <typename T, typename Context>
void doOneHotTensor(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    int num_classes,
                    phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn OneHot tensor";
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());
  // support 0D tensor
  if (x.dims().size() == 0) {
    out_dims.insert(out_dims.begin(), 1);
  }
  if (isEnvEnable("ENABLE_PARALLEL_TP")) {
    phi::DenseTensorMeta bound_meta = {x.dtype(), phi::make_ddim({1})};
    phi::DenseTensorMeta compare_meta = {phi::DataType::BOOL, x.dims()};

    phi::DenseTensor upper_bound;
    upper_bound.set_meta(bound_meta);
    dev_ctx.template Alloc<T>(&upper_bound);
    sdaa_ops::doFillTensor<T>(
        dev_ctx, static_cast<T>(num_classes), x.dtype(), &upper_bound);

    phi::DenseTensor lower_bound;
    lower_bound.set_meta(bound_meta);
    dev_ctx.template Alloc<T>(&lower_bound);
    sdaa_ops::doFillTensor<T>(
        dev_ctx, static_cast<T>(0), x.dtype(), &lower_bound);

    phi::DenseTensor x_large;
    x_large.set_meta(compare_meta);
    dev_ctx.template Alloc<bool>(&x_large);

    phi::DenseTensor x_min;
    x_min.set_meta(compare_meta);
    dev_ctx.template Alloc<bool>(&x_min);

    sdaa_ops::doCompareTensor(
        dev_ctx, x, upper_bound, CompareType::LessThan, &x_large);
    sdaa_ops::doCompareTensor(
        dev_ctx, x, lower_bound, CompareType::GreaterEqual, &x_min);

    phi::DenseTensor x_true;
    x_true.set_meta(compare_meta);
    dev_ctx.template Alloc<bool>(&x_true);

    sdaa_ops::doBitwiseBinaryOpTensor(
        dev_ctx, x_large, x_min, BitwiseOpType::And, &x_true);

    phi::DenseTensor x_true_cast;
    x_true_cast.Resize(x.dims());
    dev_ctx.Alloc(&x_true_cast, x.dtype());
    sdaa_ops::doCastTensor(dev_ctx, x_true, &x_true_cast);

    phi::DenseTensor y_true_cast;
    y_true_cast.Resize(x.dims());
    dev_ctx.Alloc(&y_true_cast, out->dtype());
    sdaa_ops::doCastTensor(dev_ctx, x_true, &y_true_cast);

    phi::DenseTensor real_x_label;
    real_x_label.Resize(x.dims());
    dev_ctx.Alloc(&real_x_label, x.dtype());
    sdaa_ops::doElementMul(dev_ctx, x, x_true_cast, -1, &real_x_label);

    std::vector<int64_t> size_vec = phi::vectorize(y_true_cast.dims());
    size_vec.push_back(1);
    y_true_cast.Resize(phi::make_ddim({size_vec}));

    phi::DenseTensor x_true_expand;
    x_true_expand.Resize(out->dims());
    dev_ctx.Alloc(&x_true_expand, out->dtype());
    sdaa_ops::doExpandTensor(dev_ctx, y_true_cast, &x_true_expand);

    phi::DenseTensor out_tmp;
    out_tmp.Resize(out->dims());
    dev_ctx.Alloc(&out_tmp, out->dtype());

    tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
    tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
        x_dims, x.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
        out_dims, out->dtype(), TensorFormat::Undefined);

    TECODNN_CHECK(tecodnnOneHot(tecodnnHandle,
                                num_classes,
                                x_Desc,
                                real_x_label.data(),
                                out_Desc,
                                out_tmp.data()));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));

    sdaa_ops::doElementMul(dev_ctx, x_true_expand, out_tmp, -1, out);
  } else {
    tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
    tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
        x_dims, x.dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
        out_dims, out->dtype(), TensorFormat::Undefined);

    TECODNN_CHECK(tecodnnOneHot(
        tecodnnHandle, num_classes, x_Desc, x.data(), out_Desc, out->data()));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
  }
}

template <typename T, typename Context>
void OneHotRawKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::Scalar& depth_scalar,
                     phi::DataType dtype,
                     bool allow_out_of_range,
                     phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA OneHotRawKernel";

  int depth = depth_scalar.to<int>();
  auto out_dims = out->dims();
  out_dims[out_dims.size() - 1] = depth;
  out->Resize(out_dims);

  dev_ctx.template Alloc<float>(out);

  custom_kernel::doOneHotTensor<T, Context>(dev_ctx, x, depth, out);
}

template <typename T, typename Context>
void OneHotKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& num_classes_s,
                  phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA OneHotKernel";

  custom_kernel::OneHotRawKernel<T, Context>(
      dev_ctx, x, num_classes_s, phi::DataType::FLOAT32, false, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(one_hot_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::OneHotRawKernel,
                          int32_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(
    one_hot, sdaa, ALL_LAYOUT, custom_kernel::OneHotKernel, int32_t, int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
}
