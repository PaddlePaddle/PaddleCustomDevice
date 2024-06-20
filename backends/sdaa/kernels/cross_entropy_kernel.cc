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

#include "kernels/funcs/high_precision_op_list.h"
#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
namespace custom_kernel {

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

template <typename T = int>
static inline T SizeToAxis(const int axis, phi::DDim dims) {
  T size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T = int>
static inline int SizeFromAxis(const int axis, phi::DDim dims) {
  T size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T = int>
static inline int SizeOutAxis(const int axis, phi::DDim dims) {
  T size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T, typename Context>
void crossEntropy(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& labels,
                  int ignore_index,
                  int axis,
                  bool soft_label,
                  phi::DenseTensor* loss) {
  auto handle = GetHandleFromCTX(dev_ctx);
  const int rank = x.dims().size();
  const int axis_v = CanonicalAxis(axis, rank);
  int axis_dim = x.dims()[axis_v];

  PADDLE_ENFORCE_GT(
      axis_dim,
      0,
      phi::errors::InvalidArgument(
          "The axis dimention should be larger than 0, but received "
          "axis dimention is %d.",
          axis_dim));

  const int n = SizeToAxis(axis_v, x.dims());
  const int d = SizeFromAxis(axis_v, x.dims());

  // weight is processsed outside the kernel, so we can't access actual weight
  phi::DenseTensor w;
  w.Resize({d});
  dev_ctx.template Alloc<T>(&w);
  sdaa_ops::doFillTensor<T>(dev_ctx, static_cast<T>(1.f), w.dtype(), &w);

  phi::DenseTensor labels_cast;
  if (soft_label) {
    PADDLE_ENFORCE_EQ(
        labels.numel(),
        x.numel(),
        phi::errors::Unimplemented(
            "The shape of labels should be equal to logits in soft label,"
            "but got size of labels is %s and phi::funcs::SizeToAxis is %s.",
            labels.dims(),
            x.dims()));
    labels_cast = labels;
  } else {
    PADDLE_ENFORCE_EQ(
        labels.numel(),
        n,
        phi::errors::Unimplemented(
            "The size of labels should be equal to phi::funcs::SizeToAxis of "
            "logits in hard label,"
            "but got size of labels is %d and phi::funcs::SizeToAxis is %d.",
            labels.numel(),
            n));
    phi::DenseTensorMeta labels_meta = {phi::DataType::INT32, labels.dims()};
    labels_cast.set_meta(labels_meta);
    dev_ctx.template Alloc<int32_t>(&labels_cast);
    sdaa_ops::doCastTensor(dev_ctx, labels, &labels_cast);
  }

  std::vector<int> label_dims;
  if (soft_label) {
    label_dims = {n, d};
  } else {
    label_dims = {n};
  }

  auto x_desc = sdaa_ops::GetTecodnnTensorDesc({n, d}, x.dtype());
  auto labels_desc =
      sdaa_ops::GetTecodnnTensorDesc(label_dims, labels_cast.dtype());
  auto loss_desc = sdaa_ops::GetTecodnnTensorDesc({n, 1}, loss->dtype());
  auto weight_desc =
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(w.dims()), w.dtype());

  TECODNN_CHECK(tecodnnCELossForward(handle,
                                     ignore_index,
                                     TECODNN_LOSS_REDUCTION_NONE,
                                     x_desc,
                                     x.data(),
                                     labels_desc,
                                     labels_cast.data(),
                                     weight_desc,
                                     w.data(),
                                     loss_desc,
                                     loss->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(labels_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(weight_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(loss_desc));
}

template <typename T, typename Context>
void crossEntropyGrad(const Context& dev_ctx,
                      const phi::DenseTensor& labels,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& loss_grad,
                      int ignore_index,
                      int axis,
                      bool soft_label,
                      phi::DenseTensor* x_grad) {
  auto handle = GetHandleFromCTX(dev_ctx);
  const int rank = x.dims().size();
  const int axis_v = CanonicalAxis(axis, rank);
  int axis_dim = x.dims()[axis_v];

  PADDLE_ENFORCE_GT(
      axis_dim,
      0,
      phi::errors::InvalidArgument(
          "The axis dimention should be larger than 0, but received "
          "axis dimention is %d.",
          axis_dim));

  const int n = SizeToAxis(axis_v, x.dims());
  const int d = SizeFromAxis(axis_v, x.dims());

  // weight is processsed outside the kernel, so we can't access actual weight
  phi::DenseTensor w;
  w.Resize({1, d});
  dev_ctx.template Alloc<T>(&w);
  sdaa_ops::doFillTensor<T>(dev_ctx, static_cast<T>(1.f), w.dtype(), &w);

  phi::DenseTensor labels_cast;
  if (soft_label) {
    PADDLE_ENFORCE_EQ(
        labels.numel(),
        x.numel(),
        phi::errors::Unimplemented(
            "The shape of labels should be equal to logits in soft label,"
            "but got size of labels is %s and phi::funcs::SizeToAxis is %s.",
            labels.dims(),
            x.dims()));
    labels_cast = labels;
  } else {
    PADDLE_ENFORCE_EQ(
        labels.numel(),
        n,
        phi::errors::Unimplemented(
            "The size of labels should be equal to phi::funcs::SizeToAxis of "
            "logits in hard label,"
            "but got size of labels is %d and phi::funcs::SizeToAxis is %d.",
            labels.numel(),
            n));
    phi::DenseTensorMeta labels_meta = {phi::DataType::INT32, labels.dims()};
    labels_cast.set_meta(labels_meta);
    dev_ctx.template Alloc<int32_t>(&labels_cast);
    sdaa_ops::doCastTensor(dev_ctx, labels, &labels_cast);
  }

  phi::DenseTensor x_2d(x), labels_d(labels_cast), loss_2d(loss_grad);
  x_2d.Resize({n, d});
  if (soft_label) {
    labels_d.Resize({n, d});
  } else {
    labels_d.Resize({labels_d.numel()});
  }

  loss_2d.Resize({n, 1});

  auto x_desc = sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(x_2d.dims()),
                                               x_2d.dtype());
  auto labels_desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(labels_d.dims()), labels_d.dtype());
  auto loss_desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(loss_2d.dims()), loss_2d.dtype());
  auto weight_desc =
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(w.dims()), w.dtype());
  TECODNN_CHECK(tecodnnCELossBackward(handle,
                                      ignore_index,
                                      TECODNN_LOSS_REDUCTION_NONE,
                                      x_desc,
                                      x.data(),
                                      labels_desc,
                                      labels_d.data(),
                                      weight_desc,
                                      w.data(),
                                      loss_desc,
                                      loss_grad.data(),
                                      x_desc,
                                      x_grad->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(labels_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(weight_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(loss_desc));
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxKernel(const Context& dev_ctx,
                                   const phi::DenseTensor& logits,
                                   const phi::DenseTensor& labels,
                                   bool soft_label,
                                   bool use_softmax,
                                   bool numeric_stable_mode,
                                   int ignore_index,
                                   int axis,
                                   phi::DenseTensor* softmax,
                                   phi::DenseTensor* loss) {
  VLOG(4) << "Call SDAA CrossEntropyWithSoftmaxKernel";

  dev_ctx.template Alloc<T>(loss);

  // input is softmax, skip softmax
  if (!use_softmax) {
    // input is softmax, just copy along
    phi::Copy(dev_ctx, logits, dev_ctx.GetPlace(), false, softmax);
    crossEntropy<T, Context>(
        dev_ctx, logits, labels, ignore_index, axis, soft_label, loss);
    return;
  }

  dev_ctx.template Alloc<T>(softmax);

  bool high_precision = false;
  if (is_in_high_precision_op_list("softmax_with_cross_entropy"))
    high_precision = true;

  sdaa_ops::doSoftmaxForward(dev_ctx, logits, axis, high_precision, softmax);
  crossEntropy<T, Context>(
      dev_ctx, *softmax, labels, ignore_index, axis, soft_label, loss);
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxGradKernel(const Context& dev_ctx,
                                       const phi::DenseTensor& labels,
                                       const phi::DenseTensor& softmax,
                                       const phi::DenseTensor& loss_grad,
                                       bool soft_label,
                                       bool use_softmax,
                                       bool numeric_stable_mode,
                                       int ignore_index,
                                       int axis,
                                       phi::DenseTensor* logits_grad) {
  VLOG(4) << "Call SDAA CrossEntropyWithSoftmaxGradKernel";

  dev_ctx.template Alloc<T>(logits_grad);

  // input is softmax, skip softmax
  if (!use_softmax) {
    phi::DenseTensor dlogits;
    // CELoss is not an inplace operator
    if (logits_grad->IsSharedWith(softmax)) {
      phi::Copy(dev_ctx, softmax, dev_ctx.GetPlace(), false, &dlogits);
    } else {
      dlogits = softmax;
    }
    crossEntropyGrad<T, Context>(dev_ctx,
                                 labels,
                                 dlogits,
                                 loss_grad,
                                 ignore_index,
                                 axis,
                                 soft_label,
                                 logits_grad);
    return;
  }

  phi::DenseTensor dlogits;
  phi::DenseTensorMeta dlogits_meta = {softmax.dtype(), softmax.dims()};
  dlogits.set_meta(dlogits_meta);
  dev_ctx.template Alloc<T>(&dlogits);

  crossEntropyGrad<T, Context>(dev_ctx,
                               labels,
                               softmax,
                               loss_grad,
                               ignore_index,
                               axis,
                               soft_label,
                               &dlogits);

  bool high_precision = false;

  if (is_in_high_precision_op_list("softmax_with_cross_entropy_grad"))
    high_precision = true;

  sdaa_ops::doSoftmaxBackward(
      dev_ctx, softmax, dlogits, axis, high_precision, logits_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxGradKernel,
                          float,
                          phi::dtype::float16) {}
