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

#include <stdio.h>

#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void AccuracyRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& inference,
                       const phi::DenseTensor& indices,
                       const phi::DenseTensor& label,
                       phi::DenseTensor* accuracy,
                       phi::DenseTensor* correct,
                       phi::DenseTensor* total) {
  VLOG(4) << "Call sdaa Accuracy kernel";
  dev_ctx.template Alloc<T>(accuracy);
  dev_ctx.template Alloc<T>(correct);
  dev_ctx.template Alloc<T>(total);

  std::vector<int32_t> label_data;
  std::vector<int32_t> indices_data;
  std::vector<T> cpuinference;

  if (label.dtype() == phi::DataType::INT64) {
    std::vector<int64_t> label_data_temp;
    TensorToVector(dev_ctx, label, dev_ctx, &label_data_temp);
    for (int i = 0; i < label.numel(); ++i) {
      label_data.push_back(static_cast<int32_t>(label_data_temp[i]));
    }
  } else if (label.dtype() == phi::DataType::INT32) {
    TensorToVector(dev_ctx, label, dev_ctx, &label_data);
  }

  if (indices.dtype() == phi::DataType::INT64) {
    std::vector<int64_t> indices_data_temp;
    TensorToVector(dev_ctx, indices, dev_ctx, &indices_data_temp);
    for (int i = 0; i < indices.numel(); ++i) {
      indices_data.push_back(static_cast<int32_t>(indices_data_temp[i]));
    }
  } else if (indices.dtype() == phi::DataType::INT32) {
    TensorToVector(dev_ctx, indices, dev_ctx, &indices_data);
  }
  TensorToVector(dev_ctx, inference, dev_ctx, &cpuinference);
  std::vector<T> num_samples_vec;
  num_samples_vec.push_back(static_cast<T>(inference.dims()[0]));
  if (num_samples_vec[0] == 0.0) {
    return;
  }
  TensorFromVector(dev_ctx, num_samples_vec, dev_ctx, total);

  size_t topk_res = inference.dims()[1];
  T num_correct = 0.0;
  T num_samples = num_samples_vec[0];
  for (size_t i = 0; i < num_samples; ++i) {
    PADDLE_ENFORCE_GE(
        label_data[i],
        0,
        phi::errors::InvalidArgument(
            "label of AccuracyOp must >= 0, But received label[%d] is %d",
            i,
            label_data[i]));
    for (size_t j = 0; j < topk_res; ++j) {
      if (indices_data[i * topk_res + j] == label_data[i]) {
        ++num_correct;
        break;
      }
    }
  }
  std::vector<T> num_correct_vec;
  num_correct_vec.push_back(num_correct);
  TensorFromVector(dev_ctx, num_correct_vec, dev_ctx, correct);
  T acc_data =
      static_cast<float>(num_correct) / static_cast<float>(num_samples);
  std::vector<T> acc_data_vec;
  acc_data_vec.push_back(acc_data);
  TensorFromVector(dev_ctx, acc_data_vec, dev_ctx, accuracy);
  accuracy->Resize(phi::make_ddim({}));
  correct->Resize(phi::make_ddim({}));
  total->Resize(phi::make_ddim({}));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    accuracy, sdaa, ALL_LAYOUT, custom_kernel::AccuracyRawKernel, float) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT64);
  kernel->InputAt(2).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
