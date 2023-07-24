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

#include "kernels.h"  //NOLINT
#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"  //NOLINT

namespace custom_kernel {

template <typename T, typename U>
void CrossEntropy(const T* prob,
                  const U* label,
                  bool soft_label,
                  size_t batch_size,
                  size_t num_classes,
                  int ignore_index,
                  int axis_dim,
                  T* out) {
  auto num_remain = num_classes / axis_dim;
  if (soft_label) {
    for (auto i = 0; i < batch_size; ++i) {
      for (auto k = 0; k < num_remain; ++k) {
        out[i * num_remain + k] = 0;
        for (auto j = 0; j < axis_dim; ++j) {
          auto idx = i * num_classes + j * num_remain + k;
          out[i * num_remain + k] -=
              label[idx] * phi::TolerableValue<T>(std::log(prob[idx]));
        }
      }
    }
  } else {
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < num_remain; j++) {
        int lbl = static_cast<int>(label[i * num_remain + j]);
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
        out[loss_idx] = lbl == ignore_index
                            ? 0
                            : -phi::TolerableValue<T>(std::log(prob[index]));
      }
    }
  }
}

template <typename T>
void CrossEntropyKernel(const phi::Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& label,
                        bool soft_label,
                        int ignore_index,
                        int axis,
                        phi::DenseTensor* out) {
  auto x_dims = x.dims();
  const int rank = x_dims.size();
  const int axis_v = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = x_dims[axis_v];

  PD_CHECK(axis_dim > 0,
           "The axis dimention should be larger than 0, but received "
           "axis dimention is %d.",
           axis_dim);

  auto out_data = dev_ctx.template Alloc<T>(out);

  const int n = phi::funcs::SizeToAxis(axis_v, x.dims());
  PD_CHECK(n > 0,
           "The size of axis should be larger than 0, but received "
           "SizeToAxis of softmax is %d.",
           n);

  const int d = phi::funcs::SizeFromAxis(axis_v, x.dims());
  if (soft_label) {
    CrossEntropy<T, T>(x.data<T>(),
                       label.data<T>(),
                       soft_label,
                       n,
                       d,
                       ignore_index,
                       axis_dim,
                       out_data);
  } else if (label.dtype() == phi::DataType::INT32) {
    CrossEntropy<T, int32_t>(x.data<T>(),
                             label.data<int32_t>(),
                             soft_label,
                             n,
                             d,
                             ignore_index,
                             axis_dim,
                             out_data);
  } else if (label.dtype() == phi::DataType::INT64) {
    CrossEntropy<T, int64_t>(x.data<T>(),
                             label.data<int64_t>(),
                             soft_label,
                             n,
                             d,
                             ignore_index,
                             axis_dim,
                             out_data);
  } else if (label.dtype() == phi::DataType::INT16) {
    CrossEntropy<T, int16_t>(x.data<T>(),
                             label.data<int16_t>(),
                             soft_label,
                             n,
                             d,
                             ignore_index,
                             axis_dim,
                             out_data);
  } else if (label.dtype() == phi::DataType::INT8) {
    CrossEntropy<T, int8_t>(x.data<T>(),
                            label.data<int8_t>(),
                            soft_label,
                            n,
                            d,
                            ignore_index,
                            axis_dim,
                            out_data);
  } else if (label.dtype() == phi::DataType::UINT8) {
    CrossEntropy<T, uint8_t>(x.data<T>(),
                             label.data<uint8_t>(),
                             soft_label,
                             n,
                             d,
                             ignore_index,
                             axis_dim,
                             out_data);
  } else {
    PD_CHECK(false, "The dtype of label must be int.");
  }
}

template <typename T>
void CrossEntropyWithSoftmaxKernel(const phi::Context& dev_ctx,
                                   const phi::DenseTensor& logits,
                                   const phi::DenseTensor& label,
                                   bool soft_label,
                                   bool use_softmax,
                                   bool numeric_stable_mode,
                                   int ignore_index,
                                   int axis,
                                   phi::DenseTensor* softmax,
                                   phi::DenseTensor* loss) {
  // do not with softmax op, and input is softmax
  if (!use_softmax) {
    auto softmax_data = dev_ctx.template Alloc<T>(softmax);
    CrossEntropyKernel<T>(
        dev_ctx, logits, label, soft_label, ignore_index, axis, loss);
    memcpy(softmax_data, logits.data<T>(), sizeof(T) * logits.numel());
    return;
  }

  custom_kernel::SoftmaxKernel<T>(dev_ctx, logits, axis, softmax);
  CrossEntropyKernel<T>(
      dev_ctx, *softmax, label, soft_label, ignore_index, axis, loss);
}

template <typename T, typename LabelT>
void CrossEntropyWithSoftmaxGradCPUKernel(const phi::Context& dev_ctx,
                                          const phi::DenseTensor& label,
                                          const phi::DenseTensor& softmax,
                                          const phi::DenseTensor& loss_grad,
                                          bool soft_label,
                                          bool use_softmax,
                                          bool numeric_stable_mode,
                                          int ignore_index,
                                          int axis,
                                          phi::DenseTensor* logits_grad) {
  const phi::DenseTensor* out_grad = &loss_grad;
  phi::DenseTensor* logit_grad = logits_grad;
  logits_grad->Resize(softmax.dims());
  dev_ctx.template Alloc<T>(logits_grad);
  auto logits_grad_data = logits_grad->data<T>();
  auto softmax_data = softmax.data<T>();

  if (logit_grad != &softmax || !use_softmax) {
    memcpy(logits_grad_data, softmax_data, softmax.numel() * sizeof(T));
  }

  const int rank = logit_grad->dims().size();
  const int axis_v = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = logit_grad->dims()[axis_v];
  PD_CHECK(axis_dim > 0,
           "The axis dimention should be larger than 0, but received "
           "axis dimention is %d.",
           axis_dim);

  const int n = phi::funcs::SizeToAxis(axis_v, logit_grad->dims());
  PD_CHECK(n > 0,
           "The size of axis should be larger than 0, but received "
           "SizeToAxis of logit_grad is %d.",
           n);

  const int d = phi::funcs::SizeFromAxis(axis_v, logit_grad->dims());
  int remain = d / axis_dim;

  auto out_grad_data = out_grad->data<T>();
  auto label_data = label.data<LabelT>();
  auto logit_grad_data = logit_grad->data<T>();
  if (!use_softmax) {
    // use_softmax step1
    if (soft_label) {
      for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < axis_dim; ++j) {
          for (auto k = 0; k < remain; ++k) {
            auto index = i * d + j * remain + k;
            auto l_index = i * remain + k;
            logit_grad_data[index] = -label_data[index] /
                                     logit_grad_data[index] *
                                     out_grad_data[l_index];
          }
        }
      }
    } else {
      // use_softmax step2
      const int remain = d / axis_dim;
      for (int i = 0; i < n; ++i) {         // for each sample_1_dim
        for (int j = 0; j < remain; j++) {  // for each sample_other_dims
          int idx = i * remain + j;  // this sample's label_idx. for 1d case,
                                     // remain=1 and j=0, so, idx = i
          auto lbl = static_cast<int64_t>(label_data[idx]);
          if (lbl == ignore_index) {
            for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
              logit_grad_data[i * d + k * remain + j] = 0;
            }
          } else {
            // only for this sample's label_idx, the label is 1, others is 0,
            // so, only compute this label_idx's class
            logit_grad_data[i * d + lbl * remain + j] =
                (-1 / logit_grad_data[i * d + lbl * remain + j]) *
                out_grad_data[idx];
            for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
              if (k !=
                  label_data[idx]) {  // label_data[idx]: this sample's label
                logit_grad_data[i * d + k * remain + j] = 0;
              }
            }
          }
        }
      }
    }
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
          logit_grad_data[index] = out_grad_data[l_index] *
                                   (logit_grad_data[index] - label_data[index]);
        }
      }
    }

    // for each sample, i is sample id
    // 1) compute dy/dx by p_j - y_j or P-Y, where j is class id,
    // P=logit_grad_mat[i] is all class's probs, Y=lbl_mat[i] is
    // all class's label
    // 2) compute dy * dy/dx by   Chain rule, dy=out_grad_mat[i]
    // for high dims, e.g. (n,c) or (n,d1,...,dm, c), compute grad by matrix
    // operation

  } else {
    for (auto i = 0; i < n; ++i) {
      for (auto j = 0; j < axis_dim; ++j) {
        for (auto k = 0; k < remain; ++k) {
          auto index = i * d + j * remain + k;
          auto l_index = i * remain + k;

          logit_grad_data[index] =
              out_grad_data[l_index] * logit_grad_data[index];
        }
      }
    }
    for (int i = 0; i < n; ++i) {         // for each sample_1_dim
      for (int j = 0; j < remain; j++) {  // for each sample_other_dims
        int idx = i * remain + j;  // this sample's label_idx. for 1d case,
                                   // remain=1 and j=0, so, idx = i
        auto lbl = static_cast<int64_t>(label_data[idx]);
        if (lbl == ignore_index) {
          for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
            logit_grad_data[i * d + k * remain + j] = 0;
          }
        } else {
          // only for this sample's label_idx, the label is 1, others is 0,
          // so, only compute this label_idx's class
          // for 1d case, remain=1 and j=0, so, [i * d + label_data[idx] *
          // remain + j] = [i * d + label_data[idx]]
          // let idx_x = i * d + label_data[idx] * remain + j,
          //   logit_grad_data[idx_x] = logit_grad_data[idx_x] -
          //   out_grad_data[idx]
          // note: logit_grad_mat = logit_grad_mat * out_grad_mat
          // so: logit_grad_data[idx_x] =  (logit_grad_data[idx_x] - 1) *
          // out_grad_data[idx]
          // means:           dy/dp * dy=   ( p - y ) * dy

          logit_grad_data[i * d + lbl * remain + j] -= out_grad_data[idx];
        }
      }
    }
  }
}

template <typename T>
void CrossEntropyWithSoftmaxGradKernel(const phi::Context& dev_ctx,
                                       const phi::DenseTensor& label,
                                       const phi::DenseTensor& softmax,
                                       const phi::DenseTensor& loss_grad,
                                       bool soft_label,
                                       bool use_softmax,
                                       bool numeric_stable_mode,
                                       int ignore_index,
                                       int axis,
                                       phi::DenseTensor* logits_grad) {
  if (soft_label) {
    CrossEntropyWithSoftmaxGradCPUKernel<T, T>(dev_ctx,
                                               label,
                                               softmax,
                                               loss_grad,
                                               soft_label,
                                               use_softmax,
                                               numeric_stable_mode,
                                               ignore_index,
                                               axis,
                                               logits_grad);
  } else if (label.dtype() == phi::DataType::INT32) {
    CrossEntropyWithSoftmaxGradCPUKernel<T, int32_t>(dev_ctx,
                                                     label,
                                                     softmax,
                                                     loss_grad,
                                                     soft_label,
                                                     use_softmax,
                                                     numeric_stable_mode,
                                                     ignore_index,
                                                     axis,
                                                     logits_grad);
  } else if (label.dtype() == phi::DataType::INT64) {
    CrossEntropyWithSoftmaxGradCPUKernel<T, int64_t>(dev_ctx,
                                                     label,
                                                     softmax,
                                                     loss_grad,
                                                     soft_label,
                                                     use_softmax,
                                                     numeric_stable_mode,
                                                     ignore_index,
                                                     axis,
                                                     logits_grad);
  } else if (label.dtype() == phi::DataType::INT16) {
    CrossEntropyWithSoftmaxGradCPUKernel<T, int16_t>(dev_ctx,
                                                     label,
                                                     softmax,
                                                     loss_grad,
                                                     soft_label,
                                                     use_softmax,
                                                     numeric_stable_mode,
                                                     ignore_index,
                                                     axis,
                                                     logits_grad);
  } else if (label.dtype() == phi::DataType::INT8) {
    CrossEntropyWithSoftmaxGradCPUKernel<T, int8_t>(dev_ctx,
                                                    label,
                                                    softmax,
                                                    loss_grad,
                                                    soft_label,
                                                    use_softmax,
                                                    numeric_stable_mode,
                                                    ignore_index,
                                                    axis,
                                                    logits_grad);
  } else if (label.dtype() == phi::DataType::UINT8) {
    CrossEntropyWithSoftmaxGradCPUKernel<T, uint8_t>(dev_ctx,
                                                     label,
                                                     softmax,
                                                     loss_grad,
                                                     soft_label,
                                                     use_softmax,
                                                     numeric_stable_mode,
                                                     ignore_index,
                                                     axis,
                                                     logits_grad);
  } else {
    PD_CHECK(false, "The dtype of label must be int.");
  }
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(cross_entropy_with_softmax,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::CrossEntropyWithSoftmaxKernel,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(cross_entropy_with_softmax_grad,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::CrossEntropyWithSoftmaxGradKernel,
                    float,
                    double) {}
