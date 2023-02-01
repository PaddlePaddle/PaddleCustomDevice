/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {
template <typename T>
T TolerableValue(const T& x) {
  const T kApproInf = 1e20;
  if (x == INFINITY) return kApproInf;
  if (x == -INFINITY) return -kApproInf;
  return x;
}

template <typename T, typename U>
void CrossEntropyCpuImpl(const T* logits,
                         const U* labels,
                         bool soft_label,
                         size_t batch_size,
                         size_t num_classes,
                         int ignore_index,
                         int axis_dim,
                         T* loss_out) {
  auto num_remain = num_classes / axis_dim;
  if (soft_label) {
    for (auto i = 0; i < batch_size; ++i) {
      for (auto k = 0; k < num_remain; ++k) {
        loss_out[i * num_remain + k] = 0;
        for (auto j = 0; j < axis_dim; ++j) {
          auto idx = i * num_classes + j * num_remain + k;
          loss_out[i * num_remain + k] -=
              labels[idx] * TolerableValue<T>(std::log(logits[idx]));
        }
      }
    }
  } else {
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < num_remain; j++) {
        int lbl = static_cast<int>(labels[i * num_remain + j]);
        if (lbl != ignore_index) {
          PD_CHECK(lbl >= 0,
                   "label value should >= 0 when label "
                   "value(%f) not equal to ignore_index(%f)",
                   lbl,
                   ignore_index);
          PD_CHECK(lbl < axis_dim,
                   "label value should less than the shape of axis dimension "
                   "when label value(%f) not equal to ignore_index(%f), But "
                   "received label value as %ld and shape of axis dimension "
                   "is %d",
                   lbl,
                   ignore_index,
                   lbl,
                   axis_dim);
        }
        int index = i * num_classes + lbl * num_remain + j;
        int loss_idx = i * num_remain + j;
        loss_out[loss_idx] = lbl == ignore_index
                                 ? 0
                                 : -TolerableValue<T>(std::log(logits[index]));
      }
    }
  }
}

template <typename T>
static void TranposeNPU(const phi::CustomContext& dev_ctx,
                        const aclrtStream& stream,
                        std::vector<int64_t>* perm,
                        const phi::DenseTensor& in,
                        phi::DenseTensor* out) {
  dev_ctx.Alloc<T>(out);
  NpuOpRunner runner;
  runner.SetType("Transpose")
      .AddInput(in)
      .AddInput(dev_ctx, std::move(*perm))
      .AddOutput(*out)
      .Run(stream);
}

static inline int SizeToAxis(const int axis, phi::DDim dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeFromAxis(const int axis, phi::DDim dims) {
  int size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
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
  auto logits_dims = logits.dims();
  const int rank = logits_dims.size();
  const int pos_axis = axis < 0 ? axis + rank : axis;
  const int axis_dim = logits_dims[pos_axis];
  const int n = SizeToAxis(pos_axis, logits_dims);
  const int d = SizeFromAxis(pos_axis, logits_dims);
  auto loss_dims = loss->dims();

  dev_ctx.template Alloc<T>(loss);
  dev_ctx.template Alloc<T>(softmax);
  auto stream = dev_ctx.stream();
  // Softmax
  if (use_softmax) {
    const auto& runner_softmax =
        NpuOpRunner("SoftmaxV2",
                    {logits},
                    {*softmax},
                    {{"axes", std::vector<int32_t>({pos_axis})}});
    runner_softmax.Run(stream);
  } else {
    // cause of input is softmax, copy to output softmax, directly
    phi::Copy<Context>(dev_ctx, logits, dev_ctx.GetPlace(), false, softmax);
  }
  if (soft_label || !use_softmax) {
    if (soft_label) {
      VLOG(4) << "soft_label = True is not supported in the npu kernel of "
                 "softmax_with_cross_entropy.";
    }
    if (!use_softmax) {
      VLOG(4) << "use_softmax = False is not supported in the npu kernel of "
                 "softmax_with_cross_entropy.";
    }
    VLOG(4) << "CrossEntropyWithSoftmaxKernel of npu is implemented using "
               "the CPU";

    std::vector<T> softmax_data_vec;
    TensorToVector(dev_ctx, *softmax, dev_ctx, &softmax_data_vec);

    phi::DenseTensor cpu_loss_out_tensor;
    cpu_loss_out_tensor.Resize({n, d / axis_dim});
    auto cpu_loss_out_data =
        dev_ctx.template HostAlloc<T>(&cpu_loss_out_tensor);
    if (soft_label) {
      std::vector<T> labels_data_vec;
      TensorToVector(dev_ctx, *&labels, dev_ctx, &labels_data_vec);
      CrossEntropyCpuImpl<T, T>(softmax_data_vec.data(),
                                labels_data_vec.data(),
                                soft_label,
                                n,
                                d,
                                ignore_index,
                                axis_dim,
                                cpu_loss_out_data);
    } else if (labels.dtype() == phi::DataType::INT32) {
      std::vector<int32_t> labels_data_vec;
      TensorToVector(dev_ctx, *&labels, dev_ctx, &labels_data_vec);
      CrossEntropyCpuImpl<T, int32_t>(softmax_data_vec.data(),
                                      labels_data_vec.data(),
                                      soft_label,
                                      n,
                                      d,
                                      ignore_index,
                                      axis_dim,
                                      cpu_loss_out_data);
    } else if (labels.dtype() == phi::DataType::INT64) {
      std::vector<int64_t> labels_data_vec;
      TensorToVector(dev_ctx, *&labels, dev_ctx, &labels_data_vec);
      CrossEntropyCpuImpl<T, int64_t>(softmax_data_vec.data(),
                                      labels_data_vec.data(),
                                      soft_label,
                                      n,
                                      d,
                                      ignore_index,
                                      axis_dim,
                                      cpu_loss_out_data);
    } else if (labels.dtype() == phi::DataType::INT16) {
      std::vector<int16_t> labels_data_vec;
      TensorToVector(dev_ctx, *&labels, dev_ctx, &labels_data_vec);
      CrossEntropyCpuImpl<T, int16_t>(softmax_data_vec.data(),
                                      labels_data_vec.data(),
                                      soft_label,
                                      n,
                                      d,
                                      ignore_index,
                                      axis_dim,
                                      cpu_loss_out_data);
    } else if (labels.dtype() == phi::DataType::INT8) {
      std::vector<int8_t> labels_data_vec;
      TensorToVector(dev_ctx, *&labels, dev_ctx, &labels_data_vec);
      CrossEntropyCpuImpl<T, int8_t>(softmax_data_vec.data(),
                                     labels_data_vec.data(),
                                     soft_label,
                                     n,
                                     d,
                                     ignore_index,
                                     axis_dim,
                                     cpu_loss_out_data);
    } else if (labels.dtype() == phi::DataType::UINT8) {
      std::vector<uint8_t> labels_data_vec;
      TensorToVector(dev_ctx, *&labels, dev_ctx, &labels_data_vec);
      CrossEntropyCpuImpl<T, uint8_t>(softmax_data_vec.data(),
                                      labels_data_vec.data(),
                                      soft_label,
                                      n,
                                      d,
                                      ignore_index,
                                      axis_dim,
                                      cpu_loss_out_data);
    } else {
      PD_CHECK(false, "The dtype of label must be int.");
    }
    TensorCopy(dev_ctx, cpu_loss_out_tensor, true, loss);
    loss->Resize(loss_dims);
  } else {
    VLOG(4) << "[DEBUG]Use NPU KERNEL";
    if (pos_axis == 1) {
      // Squeeze labels
      std::vector<int64_t> label_squeeze_shape;
      for (int32_t i = 0; i < labels.dims().size(); i++) {
        if (i != pos_axis) label_squeeze_shape.push_back(labels.dims()[i]);
      }
      phi::DenseTensor label_squeeze(labels);
      label_squeeze.Resize(phi::make_ddim(label_squeeze_shape));
      // SoftmaxCrossEntropyLoss
      phi::DenseTensor backprop;
      phi::DenseTensorMeta meta = {softmax->dtype(), softmax->dims()};
      backprop.set_meta(meta);
      dev_ctx.template Alloc<T>(&backprop);
      NpuOpRunner runner_s;
      runner_s.SetType("SoftmaxCrossEntropyLoss")
          .AddInput(logits)
          .AddInput(label_squeeze)
          .AddOutput(*loss)
          .AddOutput(backprop)
          .AddAttr("ignore_index", -1)
          .AddAttr("reduction", std::string("none"));
      runner_s.Run(stream);
    } else {
      std::vector<int64_t> perm(rank);
      for (size_t i = 0; i < rank; i++) {
        perm[i] = i;
      }
      perm[1] = pos_axis;
      perm[pos_axis] = 1;
      // Transpose logits
      std::vector<int64_t> trans_logits_shape;
      auto logits_dims = logits.dims();
      for (size_t i = 0; i < perm.size(); i++) {
        trans_logits_shape.emplace_back(logits_dims[perm[i]]);
      }
      phi::DenseTensor trans_logits;
      phi::DenseTensorMeta trans_logits_meta = {
          logits.dtype(), phi::make_ddim(trans_logits_shape)};
      trans_logits.set_meta(trans_logits_meta);
      TranposeNPU<T>(dev_ctx, stream, &perm, logits, &trans_logits);
      // Transpose labels
      std::vector<int64_t> trans_labels_shape;
      auto labels_dims = labels.dims();
      for (size_t i = 0; i < perm.size(); i++) {
        trans_labels_shape.emplace_back(labels_dims[perm[i]]);
      }
      phi::DenseTensor trans_labels;
      phi::DenseTensorMeta trans_labels_meta = {
          labels.dtype(), phi::make_ddim(trans_labels_shape)};
      trans_labels.set_meta(trans_labels_meta);
      if (labels.dtype() == phi::DataType::INT32) {
        TranposeNPU<int32_t>(dev_ctx, stream, &perm, labels, &trans_labels);
      } else if (labels.dtype() == phi::DataType::INT64) {
        TranposeNPU<int64_t>(dev_ctx, stream, &perm, labels, &trans_labels);
      } else if (labels.dtype() == phi::DataType::INT16) {
        TranposeNPU<int16_t>(dev_ctx, stream, &perm, labels, &trans_labels);
      } else if (labels.dtype() == phi::DataType::INT8) {
        TranposeNPU<int8_t>(dev_ctx, stream, &perm, labels, &trans_labels);
      } else if (labels.dtype() == phi::DataType::UINT8) {
        TranposeNPU<uint8_t>(dev_ctx, stream, &perm, labels, &trans_labels);
      }
      // Squeeze labels
      std::vector<int64_t> label_squeeze_shape;
      for (int32_t i = 0; i < trans_labels.dims().size(); i++) {
        if (i != 1) label_squeeze_shape.push_back(trans_labels.dims()[i]);
      }
      phi::DenseTensor label_squeeze(trans_labels);
      label_squeeze.Resize(phi::make_ddim(label_squeeze_shape));
      // SoftmaxCrossEntropyLoss
      phi::DenseTensor trans_loss;
      phi::DenseTensorMeta trans_loss_meta = {
          loss->dtype(), phi::make_ddim(trans_labels_shape)};
      trans_loss.set_meta(trans_loss_meta);
      dev_ctx.template Alloc<T>(&trans_loss);

      phi::DenseTensor backprop;
      phi::DenseTensorMeta backprop_meta = {logits.dtype(),
                                            phi::make_ddim(trans_logits_shape)};
      backprop.set_meta(backprop_meta);
      dev_ctx.template Alloc<T>(&backprop);
      NpuOpRunner runner_s;
      runner_s.SetType("SoftmaxCrossEntropyLoss")
          .AddInput(trans_logits)
          .AddInput(label_squeeze)
          .AddOutput(trans_loss)
          .AddOutput(backprop)
          .AddAttr("ignore_index", -1)
          .AddAttr("reduction", std::string("none"));
      runner_s.Run(stream);
      // Revert loss
      TranposeNPU<T>(dev_ctx, stream, &perm, trans_loss, loss);
    }
  }
}

template <typename T, typename Context, typename LabelT>
void CrossEntropyWithSoftmaxGradCPUKernel(const Context& dev_ctx,
                                          const phi::DenseTensor& label,
                                          const phi::DenseTensor& softmax,
                                          const phi::DenseTensor& loss_grad,
                                          bool soft_label,
                                          bool use_softmax,
                                          bool numeric_stable_mode,
                                          int ignore_index,
                                          int axis,
                                          phi::DenseTensor* logits_grad) {
  auto logits_grad_dims = logits_grad->dims();
  dev_ctx.template Alloc<T>(logits_grad);
  phi::DenseTensor cpu_logits_grad_tensor;
  cpu_logits_grad_tensor.Resize(logits_grad_dims);
  auto cpu_logits_grad_data =
      dev_ctx.template HostAlloc<T>(&cpu_logits_grad_tensor);

  if (logits_grad != &softmax || !use_softmax) {
    std::vector<T> softmax_data_vec;
    TensorToVector(dev_ctx, *&softmax, dev_ctx, &softmax_data_vec);
    memcpy(cpu_logits_grad_data,
           softmax_data_vec.data(),
           softmax.numel() * sizeof(T));
  }

  const int rank = logits_grad_dims.size();
  const int axis_v = axis < 0 ? axis + rank : axis;
  int axis_dim = logits_grad_dims[axis_v];
  PD_CHECK(axis_dim > 0,
           "The axis dimention should be larger than 0, but received "
           "axis dimention is %d.",
           axis_dim);

  const int n = SizeToAxis(axis_v, logits_grad_dims);
  PD_CHECK(n > 0,
           "The size of axis should be larger than 0, but received "
           "SizeToAxis of logit_grad is %d.",
           n);

  const int d = SizeFromAxis(axis_v, logits_grad_dims);
  int remain = d / axis_dim;

  std::vector<T> out_grad_vec;
  std::vector<LabelT> label_data_vec;
  TensorToVector(dev_ctx, *&loss_grad, dev_ctx, &out_grad_vec);
  TensorToVector(dev_ctx, *&label, dev_ctx, &label_data_vec);

  if (!use_softmax) {
    // use_softmax step1
    if (soft_label) {
      for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < axis_dim; ++j) {
          for (auto k = 0; k < remain; ++k) {
            auto index = i * d + j * remain + k;
            auto l_index = i * remain + k;
            cpu_logits_grad_data[index] = -label_data_vec[index] /
                                          cpu_logits_grad_data[index] *
                                          out_grad_vec[l_index];
          }
        }
      }
    } else {
      // use_softmax step2
      for (int i = 0; i < n; ++i) {         // for each sample_1_dim
        for (int j = 0; j < remain; j++) {  // for each sample_other_dims
          int idx = i * remain + j;
          auto lbl = static_cast<int64_t>(label_data_vec[idx]);
          if (lbl == ignore_index) {
            for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
              cpu_logits_grad_data[i * d + k * remain + j] = 0;
            }
          } else {
            // only for this sample's label_idx, the label is 1, others is 0,
            // so, only compute this label_idx's class
            cpu_logits_grad_data[i * d + lbl * remain + j] =
                (-1 / cpu_logits_grad_data[i * d + lbl * remain + j]) *
                out_grad_vec[idx];
            for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
              if (k != label_data_vec[idx]) {     // label_data_vec[idx]: this
                                                  // sample's label
                cpu_logits_grad_data[i * d + k * remain + j] = 0;
              }
            }
          }
        }
      }
    }
    TensorCopy(dev_ctx, cpu_logits_grad_tensor, true, logits_grad);
    return;
  }
  // for use_softmax=False, continue
  if (soft_label) {
    // when soft_label = True, ignore_index is not supported
    for (auto i = 0; i < n; ++i) {
      for (auto j = 0; j < axis_dim; ++j) {
        for (auto k = 0; k < remain; ++k) {
          auto index = i * d + j * remain + k;
          auto l_index = i * remain + k;
          cpu_logits_grad_data[index] =
              out_grad_vec[l_index] *
              (cpu_logits_grad_data[index] - label_data_vec[index]);
        }
      }
    }
  } else {
    for (auto i = 0; i < n; ++i) {
      for (auto j = 0; j < axis_dim; ++j) {
        for (auto k = 0; k < remain; ++k) {
          auto index = i * d + j * remain + k;
          auto l_index = i * remain + k;

          cpu_logits_grad_data[index] =
              out_grad_vec[l_index] * cpu_logits_grad_data[index];
        }
      }
    }
    for (int i = 0; i < n; ++i) {         // for each sample_1_dim
      for (int j = 0; j < remain; j++) {  // for each sample_other_dims
        int idx = i * remain + j;
        auto lbl = static_cast<int64_t>(label_data_vec[idx]);
        if (lbl == ignore_index) {
          for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
            cpu_logits_grad_data[i * d + k * remain + j] = 0;
          }
        } else {
          cpu_logits_grad_data[i * d + lbl * remain + j] -= out_grad_vec[idx];
        }
      }
    }
  }
  TensorCopy(dev_ctx, cpu_logits_grad_tensor, true, logits_grad);
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
  const int rank = softmax.dims().size();
  const int pos_axis = axis < 0 ? axis + rank : axis;
  int cls_num = softmax.dims()[pos_axis];
  auto stream = dev_ctx.stream();
  if (soft_label || !use_softmax) {
    if (soft_label) {
      VLOG(4) << "soft_label = True is not supported in the npu kernel of "
                 "、.";
    }
    if (!use_softmax) {
      VLOG(4) << "use_softmax = False is not supported in the npu kernel of "
                 "softmax_with_cross_entropy.";
    }
    VLOG(4) << "CrossEntropyWithSoftmaxGradKernel of npu is implemented using "
               "the CPU";
    if (soft_label) {
      CrossEntropyWithSoftmaxGradCPUKernel<T, Context, T>(dev_ctx,
                                                          labels,
                                                          softmax,
                                                          loss_grad,
                                                          soft_label,
                                                          use_softmax,
                                                          numeric_stable_mode,
                                                          ignore_index,
                                                          axis,
                                                          logits_grad);
    } else if (labels.dtype() == phi::DataType::INT32) {
      CrossEntropyWithSoftmaxGradCPUKernel<T, Context, int32_t>(
          dev_ctx,
          labels,
          softmax,
          loss_grad,
          soft_label,
          use_softmax,
          numeric_stable_mode,
          ignore_index,
          axis,
          logits_grad);
    } else if (labels.dtype() == phi::DataType::INT64) {
      CrossEntropyWithSoftmaxGradCPUKernel<T, Context, int64_t>(
          dev_ctx,
          labels,
          softmax,
          loss_grad,
          soft_label,
          use_softmax,
          numeric_stable_mode,
          ignore_index,
          axis,
          logits_grad);
    } else if (labels.dtype() == phi::DataType::INT16) {
      CrossEntropyWithSoftmaxGradCPUKernel<T, Context, int16_t>(
          dev_ctx,
          labels,
          softmax,
          loss_grad,
          soft_label,
          use_softmax,
          numeric_stable_mode,
          ignore_index,
          axis,
          logits_grad);
    } else if (labels.dtype() == phi::DataType::INT8) {
      CrossEntropyWithSoftmaxGradCPUKernel<T, Context, int8_t>(
          dev_ctx,
          labels,
          softmax,
          loss_grad,
          soft_label,
          use_softmax,
          numeric_stable_mode,
          ignore_index,
          axis,
          logits_grad);
    } else if (labels.dtype() == phi::DataType::UINT8) {
      CrossEntropyWithSoftmaxGradCPUKernel<T, Context, uint8_t>(
          dev_ctx,
          labels,
          softmax,
          loss_grad,
          soft_label,
          use_softmax,
          numeric_stable_mode,
          ignore_index,
          axis,
          logits_grad);
    } else {
      PD_CHECK(false, "The dtype of labels must be int.");
    }
  } else {
    // cast label from int64/int32 to int32 for OneHotD
    phi::DenseTensor casted_labels;
    if (labels.dtype() != phi::DataType::INT32) {
      phi::DenseTensorMeta casted_labels_meta = {phi::DataType::INT32,
                                                 labels.dims()};
      casted_labels.set_meta(casted_labels_meta);
      dev_ctx.template Alloc<int32_t>(&casted_labels);
      auto dst_dtype = ConvertToNpuDtype(phi::DataType::INT32);
      const auto& runner_cast_label =
          NpuOpRunner("Cast",
                      {labels},
                      {casted_labels},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_label.Run(stream);
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
    FillNpuTensorWithConstant<int32_t>(
        &on_tensor, dev_ctx, static_cast<int>(1));
    FillNpuTensorWithConstant<int32_t>(
        &off_tensor, dev_ctx, static_cast<int>(0));

    // one_hot
    phi::DenseTensor tmp_onehot;
    phi::DenseTensorMeta tmp_onehot_meta = {on_tensor.dtype(), softmax.dims()};
    tmp_onehot.set_meta(tmp_onehot_meta);
    dev_ctx.template Alloc<int32_t>(&tmp_onehot);

    NpuOpRunner runner_onehot;
    runner_onehot.SetType("OneHot")
        .AddInput(casted_labels)
        .AddInput(dev_ctx,
                  std::vector<int32_t>(1, static_cast<int32_t>(cls_num)))
        .AddInput(on_tensor)
        .AddInput(off_tensor)
        .AddOutput(tmp_onehot)
        .AddAttr("axis", pos_axis);
    runner_onehot.Run(stream);

    // cast one_hot from int32 to T
    phi::DenseTensor casted_onehot;
    if (softmax.dtype() != phi::DataType::INT32) {
      phi::DenseTensorMeta onehot_meta = {softmax.dtype(), tmp_onehot.dims()};
      casted_onehot.set_meta(onehot_meta);
      dev_ctx.template Alloc<T>(&casted_onehot);
      auto dst_dtype = ConvertToNpuDtype(softmax.dtype());
      const auto& runner_cast_onehot =
          NpuOpRunner("Cast",
                      {tmp_onehot},
                      {casted_onehot},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_onehot.Run(stream);
    } else {
      casted_onehot = tmp_onehot;
    }

    // sub
    phi::DenseTensor tmp_sub;
    phi::DenseTensorMeta tmp_sub_meta = {softmax.dtype(), softmax.dims()};
    tmp_sub.set_meta(tmp_sub_meta);
    dev_ctx.template Alloc<T>(&tmp_sub);

    const auto& runner_sub =
        NpuOpRunner("Sub", {softmax, casted_onehot}, {tmp_sub}, {});
    runner_sub.Run(stream);

    // mul
    dev_ctx.template Alloc<T>(logits_grad);
    const auto& runner_mul =
        NpuOpRunner("Mul", {loss_grad, tmp_sub}, {*logits_grad}, {});
    runner_mul.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxKernel,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxGradKernel,
                          float) {}
