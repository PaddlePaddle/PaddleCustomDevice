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

#pragma once

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"
#include "kernels/funcs/reduce_op.h"

namespace custom_kernel {
template <typename T>
T TolerableValue(const T& x) {
  const T kApproInf = 1e20;
  if (x == INFINITY) return kApproInf;
  if (x == -INFINITY) return -kApproInf;
  return x;
}

template <typename T>
class CrossEntropyFunctor {
 public:
  void operator()(const Context& dev_ctx,
                  phi::DenseTensor* out,
                  const phi::DenseTensor& prob,
                  const phi::DenseTensor& labels,
                  const bool softLabel,
                  const int ignore_index,
                  const int axis_dim) {
    const int batch_size = prob.dims()[0];
    const int num_classes = prob.dims()[1];
    const int num_remain = num_classes / axis_dim;

    PADDLE_ENFORCE_EQ(prob.dims().size(),
                      2,
                      phi::errors::InvalidArgument(
                          "Two dimension input [NxC] is allowed only."));

    if (softLabel) {
      VLOG(5)
          << "[CrossEntropyFunctor] softlabel, do -(log + mul + reduce_sum)";
      Tensor log_x;
      log_x.Resize(prob.dims());
      dev_ctx.template Alloc<T>(&log_x);

      MLUCnnlTensorDesc prob_desc(prob);
      MLUCnnlTensorDesc labels_desc(labels);
      // do log, MulAx and reduce_sum along axis 1
      // 1. log, note value validation is not considered here yet.
      // @Chenxiao, take care of the value
      const cnnlComputationPreference_t prefer =
          CNNL_COMPUTATION_HIGH_PRECISION;
      MLUCnnl::Log(dev_ctx,
                   prefer,
                   CNNL_LOG_E,
                   prob_desc.get(),
                   GetBasePtr(&prob),
                   prob_desc.get(),
                   GetBasePtr(&log_x));
      // 2. mul
      MLUCnnl::MulAx(dev_ctx,
                     labels_desc.get(),
                     GetBasePtr(&labels),
                     prob_desc.get(),
                     GetBasePtr(&log_x));
      // 3. reduce_sum, note to do reshape before
      log_x.Resize({batch_size, axis_dim, num_remain});
      MLUReduceOp<T>(dev_ctx, log_x, {1}, false, false, "reduce_sum", out);
      // 4. negative
      MLUCnnlTensorDesc out_desc(*out);
      MLUCnnl::Neg(dev_ctx,
                   out_desc.get(),
                   GetBasePtr(out),
                   out_desc.get(),
                   GetBasePtr(out));
    } else {
      Tensor labels_int32;
      // cast labels to int32 dtype.
      if (labels.dtype() == DataType::INT32) {
        labels_int32 = labels;
      } else {
        labels_int32.Resize(labels.dims());
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

      // copy probs and labels to cpu and compute
      std::vector<T> prob_data_vec;
      std::vector<int32_t> labels_data_vec;
      phi::DenseTensor cpu_loss_out_tensor;
      cpu_loss_out_tensor.Resize(out->dims());
      auto cpu_loss_out_data =
          dev_ctx.template HostAlloc<T>(&cpu_loss_out_tensor);
      TensorToVector(dev_ctx, prob, dev_ctx, &prob_data_vec);
      TensorToVector(dev_ctx, labels_int32, dev_ctx, &labels_data_vec);
      dev_ctx.Wait();
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_remain; j++) {
          int lbl = static_cast<int>(labels_data_vec[i * num_remain + j]);
          if (lbl != ignore_index) {
            PADDLE_ENFORCE_GE(lbl,
                              0,
                              "label value should >= 0 when label "
                              "value(%f) not equal to ignore_index(%f)",
                              lbl,
                              ignore_index);
            PADDLE_ENFORCE_LT(
                lbl,
                axis_dim,
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
          // use float dtype to do log
          cpu_loss_out_data[loss_idx] =
              lbl == ignore_index
                  ? 0
                  : -TolerableValue<float>(
                        std::log(static_cast<float>(prob_data_vec[index])));
        }
      }

      // copy loss back to mlu out
      TensorCopy(dev_ctx, cpu_loss_out_tensor, true, out);
      out->Resize(out->dims());
    }
  }
};

}  // namespace custom_kernel
