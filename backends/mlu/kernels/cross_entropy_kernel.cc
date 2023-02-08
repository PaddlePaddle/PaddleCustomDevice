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
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

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
  phi::DenseTensor backprop;
  PADDLE_ENFORCE_EQ(use_softmax,
                    true,
                    phi::errors::InvalidArgument(
                        "use_softmax=False is not supported in "
                        "the mlu kernel of softmax_with_cross_entropy."));

  const int rank = logits.dims().size();
  axis = custom_kernel::CanonicalAxis(axis, rank);
  dev_ctx.template Alloc<T>(loss);
  dev_ctx.template Alloc<T>(softmax);

  // cnnl softmax only support 3-dims, regard all shape as [d1, d2, d3]
  const int cnnl_softmax_dims = 3;
  const int d1 = custom_kernel::SizeToAxis(axis, logits.dims());
  const int d2_logits = logits.dims()[axis];
  const int d2_labels = labels.dims()[axis];
  const int d3 = custom_kernel::SizeOutAxis(axis, logits.dims());

  // CNNL_SOFTMAX_MODE_LOW_DIMENSION has better perfermence, use it as much as
  // possible.
  cnnlSoftmaxMode_t mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
  std::vector<int> regard_logits_shape{d1, 1, d2_logits};
  std::vector<int> regard_labels_shape{d1, 1, d2_labels};
  std::vector<int> regard_loss_shape{d1, 1, 1};
  if (d3 != 1) {
    mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
    regard_logits_shape = {d1, d2_logits, d3};
    regard_labels_shape = {d1, d2_labels, d3};
    regard_loss_shape = {d1, 1, d3};
  }
  phi::DenseTensorMeta meta = {logits.dtype(), {softmax->dims()}};
  backprop.set_meta(meta);
  dev_ctx.template Alloc<T>(&backprop);

  MLUCnnlTensorDesc logits_desc(
      cnnl_softmax_dims, regard_logits_shape.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc labels_desc(
      cnnl_softmax_dims, regard_labels_shape.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc loss_desc(
      cnnl_softmax_dims, regard_loss_shape.data(), ToCnnlDataType<T>());

  const cnnlSoftmaxAlgorithm_t algo = CNNL_SOFTMAX_ACCURATE;
  MLUCnnl::SoftmaxForward(dev_ctx,
                          algo,
                          mode,
                          NULL,
                          logits_desc.get(),
                          GetBasePtr(&logits),
                          NULL,
                          logits_desc.get(),
                          GetBasePtr(softmax));

  if (soft_label) {
    VLOG(5) << "[cross_entropy] soft_label";
    const cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
    MLUCnnl::SoftmaxCrossEntropyWithLogits(dev_ctx,
                                           mode,
                                           prefer,
                                           logits_desc.get(),
                                           GetBasePtr(&logits),
                                           labels_desc.get(),
                                           GetBasePtr(&labels),
                                           loss_desc.get(),
                                           GetBasePtr(loss),
                                           logits_desc.get(),
                                           GetBasePtr(&backprop));
  } else {
    VLOG(5) << "[cross_entropy] hard_label."
            << " If d3 != 1, we need to do transpose for inputs and outputs"
            << " so that we can use CNNL_SOFTMAX_MODE_LOW_DIMENSION mode.";

    VLOG(5) << "[cross_entropy] labels dims: " << labels.dims();

    mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
    Tensor labels_int32;
    labels_int32.Resize(labels.dims());

    if (labels.dtype() == DataType::INT32) {
      labels_int32 = labels;
    } else {
      // do cast since cnnl only supports int32 labels in sparse celoss.
      dev_ctx.template Alloc<int32_t>(&labels_int32);
      MLUCnnlTensorDesc labels_desc(labels);
      MLUCnnlTensorDesc labels_int32_desc(labels_int32);
      cnnlCastDataType_t cast_type =
          GetCastDataType(labels.dtype(), DataType::INT32);
      MLUCnnl::Cast(dev_ctx,
                    cast_type,
                    labels_desc.get(),
                    GetBasePtr(&labels),
                    labels_int32_desc.get(),
                    GetBasePtr(&labels_int32));
    }

    // transpose logits, labels, loss and backprop if d3 != 1
    Tensor trans_logits, trans_labels, trans_loss;
    if (d3 == 1) {
      trans_logits = logits;
      trans_labels = labels_int32;
      trans_loss = *loss;
      trans_logits.Resize({d1, d2_logits});
      trans_labels.Resize({d1});
      trans_loss.Resize({d1});
    } else {
      VLOG(5) << "[cross_entropy] d3 != 1, do transpose to [d1, d3, d2_xxx]";
      Tensor logits_3d = logits;
      logits_3d.Resize({d1, d2_logits, d3});
      labels_int32.Resize({d1, 1, d3});
      trans_loss.Resize({d1, d3, 1});
      dev_ctx.template Alloc<T>(&trans_loss);
      std::vector<int> perm{0, 2, 1};
      TransposeFromMLUTensor<T>(dev_ctx, perm, &logits_3d, &trans_logits, true);
      TransposeFromMLUTensor<int32_t>(
          dev_ctx, perm, &labels_int32, &trans_labels, true);
      trans_labels.Resize({d1, d3});
    }

    MLUCnnlTensorDesc trans_logits_desc(trans_logits);
    MLUCnnlTensorDesc trans_labels_desc(trans_labels);
    MLUCnnlTensorDesc trans_loss_desc(trans_loss);

    // mask label with ignore_index, do with the following
    // 1. logic: mask = (label == ignore_index)
    Tensor ignore_idx_tensor, mask_tensor;
    std::vector<int64_t> ignore_dim_vec(trans_labels.dims().size(), 1);
    ignore_idx_tensor.Resize(
        phi::DDim(ignore_dim_vec.data(), ignore_dim_vec.size()));
    mask_tensor.Resize(trans_labels.dims());
    dev_ctx.template Alloc<int32_t>(&ignore_idx_tensor);
    dev_ctx.template Alloc<bool>(&mask_tensor);
    FillMLUTensorWithHostValue<int32_t>(
        dev_ctx, ignore_index, &ignore_idx_tensor);
    MLUCnnlTensorDesc ignore_index_desc(ignore_idx_tensor);
    MLUCnnlTensorDesc mask_desc(mask_tensor);
    MLUCnnl::Logic(dev_ctx,
                   CNNL_LOGIC_OP_EQ,
                   trans_labels_desc.get(),
                   GetBasePtr(&trans_labels),
                   ignore_index_desc.get(),
                   GetBasePtr(&ignore_idx_tensor),
                   mask_desc.get(),
                   GetBasePtr(&mask_tensor));
    // 2. mask: if mask = True, set 0
    int fill_scale = 0;
    MLUCnnl::Mask(dev_ctx,
                  CNNL_MASKED_FILL_HOST,
                  trans_labels_desc.get(),
                  GetBasePtr(&trans_labels),
                  mask_desc.get(),
                  GetBasePtr(&mask_tensor),
                  nullptr, /*value_desc*/
                  nullptr, /*value*/
                  &fill_scale,
                  trans_labels_desc.get(),
                  GetBasePtr(&trans_labels),
                  nullptr /*number*/);

    MLUCnnl::SparseSoftmaxXentWithLogits(dev_ctx,
                                         mode,
                                         trans_logits_desc.get(),
                                         GetBasePtr(&trans_logits),
                                         trans_labels_desc.get(),
                                         GetBasePtr(&trans_labels),
                                         trans_loss_desc.get(),
                                         GetBasePtr(&trans_loss),
                                         trans_logits_desc.get(),
                                         GetBasePtr(&backprop));

    if (d3 != 1) {
      VLOG(5) << "[cross_entropy] d3 != 1, transpose loss back."
              << " [d1, d3, 1] -> [d1, 1, d3] -> original shape";
      std::vector<int> perm{0, 2, 1};
      phi::DDim original_loss_dim = loss->dims();
      loss->Resize({d1, 1, d3});
      TransposeFromMLUTensor<T>(dev_ctx, perm, &trans_loss, loss, false);
      loss->Resize(original_loss_dim);
    }
  }
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
  int cls_num = softmax.dims()[softmax.dims().size() - 1];

  // cast label from int64/int32 to int32 for OneHotD
  phi::DenseTensor casted_labels;
  if (labels.dtype() != phi::DataType::INT32) {
    phi::DenseTensorMeta casted_labels_meta = {phi::DataType::INT32,
                                               labels.dims()};
    casted_labels.set_meta(casted_labels_meta);
    dev_ctx.template Alloc<int32_t>(&casted_labels);

    MLUCnnlTensorDesc labels_desc(labels);
    MLUCnnlTensorDesc casted_labels_desc(casted_labels);

    cnnlCastDataType_t cast_type =
        GetCastDataType(labels.dtype(), DataType::INT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  labels_desc.get(),
                  GetBasePtr(&labels),
                  casted_labels_desc.get(),
                  GetBasePtr(&casted_labels));
  } else {
    casted_labels = labels;
  }

  // on and off
  phi::DenseTensor on_tensor, off_tensor;
  phi::DenseTensorMeta on_off_meta = {phi::DataType::INT32, {1}};
  on_tensor.set_meta(on_off_meta);
  off_tensor.set_meta(on_off_meta);
  dev_ctx.template Alloc<int32_t>(&on_tensor);
  dev_ctx.template Alloc<int32_t>(&off_tensor);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<int>(1), &on_tensor);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<int>(0), &off_tensor);

  // one_hot
  phi::DenseTensor tmp_onehot;
  phi::DenseTensorMeta tmp_onehot_meta = {on_tensor.dtype(), softmax.dims()};
  tmp_onehot.set_meta(tmp_onehot_meta);
  dev_ctx.template Alloc<int32_t>(&tmp_onehot);
  MLUCnnlTensorDesc casted_labels_desc(casted_labels);
  MLUCnnlTensorDesc tmp_onehot_desc(tmp_onehot);
  MLUCnnl::OneHot(dev_ctx,
                  casted_labels_desc.get(),
                  GetBasePtr(&casted_labels),
                  cls_num,
                  GetBasePtr(&on_tensor),
                  GetBasePtr(&off_tensor),
                  -1,
                  CNNL_DTYPE_INT32,
                  GetBasePtr(&tmp_onehot));

  // cast one_hot from int32 to T
  phi::DenseTensor casted_onehot;
  if (softmax.dtype() != phi::DataType::INT32) {
    phi::DenseTensorMeta onehot_meta = {softmax.dtype(), tmp_onehot.dims()};
    casted_onehot.set_meta(onehot_meta);
    dev_ctx.template Alloc<T>(&casted_onehot);
    cnnlCastDataType_t cast_type =
        GetCastDataType(DataType::INT32, softmax.dtype());
    MLUCnnlTensorDesc casted_onehot_desc(casted_onehot);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  tmp_onehot_desc.get(),
                  GetBasePtr(&tmp_onehot),
                  casted_onehot_desc.get(),
                  GetBasePtr(&casted_onehot));
  } else {
    casted_onehot = tmp_onehot;
  }

  // sub
  phi::DenseTensor tmp_sub;
  phi::DenseTensorMeta tmp_sub_meta = {softmax.dtype(), softmax.dims()};
  tmp_sub.set_meta(tmp_sub_meta);
  dev_ctx.template Alloc<T>(&tmp_sub);
  MLUCnnlOpTensorDesc mul_sub_op_desc(
      CNNL_OP_TENSOR_SUB, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
  MLUCnnlTensorDesc softmax_desc(softmax);
  MLUCnnlTensorDesc tmp_sub_desc(tmp_sub);
  MLUCnnlTensorDesc casted_onehot_desc(casted_onehot);
  MLUCnnl::OpTensor(dev_ctx,
                    mul_sub_op_desc.get(),
                    softmax_desc.get(),
                    GetBasePtr(&softmax),
                    casted_onehot_desc.get(),
                    GetBasePtr(&casted_onehot),
                    tmp_sub_desc.get(),
                    GetBasePtr(&tmp_sub),
                    ToCnnlDataType<T>());

  dev_ctx.template Alloc<T>(logits_grad);
  MLUCnnlOpTensorDesc mul_mul_op_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
  MLUCnnlTensorDesc loss_grad_desc(loss_grad);
  MLUCnnlTensorDesc logits_grad_desc(*logits_grad);
  MLUCnnl::OpTensor(dev_ctx,
                    mul_mul_op_desc.get(),
                    tmp_sub_desc.get(),
                    GetBasePtr(&tmp_sub),
                    loss_grad_desc.get(),
                    GetBasePtr(&loss_grad),
                    logits_grad_desc.get(),
                    GetBasePtr(logits_grad),
                    ToCnnlDataType<T>());
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxGradKernel,
                          float,
                          phi::dtype::float16) {}
