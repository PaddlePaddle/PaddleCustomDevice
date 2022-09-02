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

namespace custom_kernel {

template <typename T, typename Context>
void AccuracyRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& indices,
                       const phi::DenseTensor& label,
                       phi::DenseTensor* accuracy,
                       phi::DenseTensor* correct,
                       phi::DenseTensor* total) {
  int num_samples = indices.dims()[0];
  if (num_samples == 0) {
    return;
  }

  // cast `indices` or `label` if their type is not INT32
  Tensor indices_int32;
  Tensor label_int32;
  auto indices_type = indices.dtype();
  if (indices_type != DataType::INT32) {
    PADDLE_ENFORCE_EQ(MLUSupportsCast(indices_type, DataType::INT32),
                      true,
                      phi::errors::Unimplemented(
                          "In accuracy mlu kernel, cast indices from [%s] to "
                          "[%s] is not supported.",
                          indices_type,
                          DataType::INT32));
    indices_int32.Resize(indices.dims());
    dev_ctx.template Alloc<int>(&indices_int32);
    MLUCnnlTensorDesc org_indices_desc(indices);
    MLUCnnlTensorDesc indices_int32_desc(indices_int32);
    cnnlCastDataType_t cast_type =
        GetCastDataType(indices_type, DataType::INT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  org_indices_desc.get(),
                  GetBasePtr(&indices),
                  indices_int32_desc.get(),
                  GetBasePtr(&indices_int32));
  } else {
    indices_int32 = indices;
  }
  auto label_type = label.dtype();
  if (label_type != DataType::INT32) {
    PADDLE_ENFORCE_EQ(
        MLUSupportsCast(label_type, DataType::INT32),
        true,
        phi::errors::Unimplemented(
            "In accuracy mlu kernel, cast label from [%s] to [%s] "
            "is not supported.",
            label_type,
            DataType::INT32));
    label_int32.Resize(label.dims());
    dev_ctx.template Alloc<int>(&label_int32);
    MLUCnnlTensorDesc org_label_desc(label);
    MLUCnnlTensorDesc label_int32_desc(label_int32);
    cnnlCastDataType_t cast_type = GetCastDataType(label_type, DataType::INT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  org_label_desc.get(),
                  GetBasePtr(&label),
                  label_int32_desc.get(),
                  GetBasePtr(&label_int32));
  } else {
    label_int32 = label;
  }

  // equal
  MLUCnnlTensorDesc indices_int32_desc(indices_int32);
  MLUCnnlTensorDesc label_int32_desc(label_int32);
  Tensor equal_tensor;
  equal_tensor.Resize(indices.dims());
  dev_ctx.template Alloc<bool>(&equal_tensor);
  MLUCnnlTensorDesc equal_tensor_desc(equal_tensor);
  MLUCnnl::Logic(dev_ctx,
                 CNNL_LOGIC_OP_EQ,
                 indices_int32_desc.get(),
                 GetBasePtr(&indices_int32),
                 label_int32_desc.get(),
                 GetBasePtr(&label_int32),
                 equal_tensor_desc.get(),
                 GetBasePtr(&equal_tensor));

  // cast equal
  Tensor equal_fp32;
  equal_fp32.Resize(indices.dims());
  dev_ctx.template Alloc<float>(&equal_fp32);
  MLUCnnlTensorDesc equal_fp32_desc(equal_fp32);
  cnnlCastDataType_t equal_cast_type =
      GetCastDataType(DataType::BOOL, DataType::FLOAT32);
  MLUCnnl::Cast(dev_ctx,
                equal_cast_type,
                equal_tensor_desc.get(),
                GetBasePtr(&equal_tensor),
                equal_fp32_desc.get(),
                GetBasePtr(&equal_fp32));

  // [correct]
  // reduce_max
  Tensor correct_max;
  correct_max.Resize(phi::make_ddim({num_samples}));
  dev_ctx.template Alloc<float>(&correct_max);
  MLUCnnlTensorDesc correct_max_desc(correct_max);
  MLUCnnlReduceDesc reduce_max_desc({1},
                                    CNNL_REDUCE_MAX,
                                    ToCnnlDataType<float>(),
                                    CNNL_NOT_PROPAGATE_NAN,
                                    CNNL_REDUCE_NO_INDICES,
                                    CNNL_32BIT_INDICES);
  MLUCnnl::Reduce(dev_ctx,
                  true /*need_workspace*/,
                  reduce_max_desc.get(),
                  nullptr,
                  equal_fp32_desc.get(),
                  GetBasePtr(&equal_fp32),
                  0 /*indices_size*/,
                  nullptr,
                  nullptr,
                  correct_max_desc.get(),
                  GetBasePtr(&correct_max));

  // reduce_sum
  Tensor correct_sum;
  correct_sum.Resize(correct->dims());
  dev_ctx.template Alloc<float>(&correct_sum);
  MLUCnnlTensorDesc correct_sum_desc(correct_sum);
  MLUCnnlReduceDesc reduce_sum_desc({0},
                                    CNNL_REDUCE_ADD,
                                    ToCnnlDataType<float>(),
                                    CNNL_NOT_PROPAGATE_NAN,
                                    CNNL_REDUCE_NO_INDICES,
                                    CNNL_32BIT_INDICES);
  MLUCnnl::Reduce(dev_ctx,
                  true /*need_workspace*/,
                  reduce_sum_desc.get(),
                  nullptr,
                  correct_max_desc.get(),
                  GetBasePtr(&correct_max),
                  0 /*indices_size*/,
                  nullptr,
                  nullptr,
                  correct_sum_desc.get(),
                  GetBasePtr(&correct_sum));

  // cast to int
  dev_ctx.template Alloc<int>(correct);
  MLUCnnlTensorDesc correct_desc(*correct);
  cnnlCastDataType_t correct_cast_type =
      GetCastDataType(DataType::FLOAT32, DataType::INT32);
  MLUCnnl::Cast(dev_ctx,
                correct_cast_type,
                correct_sum_desc.get(),
                GetBasePtr(&correct_sum),
                correct_desc.get(),
                GetBasePtr(correct));

  // [total]
  dev_ctx.template Alloc<int>(total);
  MLUCnnlTensorDesc total_desc(*total);
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &num_samples,
                total_desc.get(),
                GetBasePtr(total));

  // use `total` of type `float32` for calculating accuracy
  Tensor total_fp32;
  total_fp32.Resize(total->dims());
  dev_ctx.template Alloc<float>(&total_fp32);
  MLUCnnlTensorDesc total_fp32_desc(total_fp32);
  float num_samples_fp32 = static_cast<float>(num_samples);
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &num_samples_fp32,
                total_fp32_desc.get(),
                GetBasePtr(&total_fp32));

  // [accuracy]
  dev_ctx.template Alloc<float>(accuracy);
  MLUCnnlTensorDesc accuracy_desc(*accuracy);
  MLUCnnl::Div(dev_ctx,
               CNNL_COMPUTATION_HIGH_PRECISION,
               correct_sum_desc.get(),
               GetBasePtr(&correct_sum),
               total_fp32_desc.get(),
               GetBasePtr(&total_fp32),
               accuracy_desc.get(),
               GetBasePtr(accuracy));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(accuracy,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::AccuracyRawKernel,
                          float,
                          phi::dtype::float16,
                          int,
                          int64_t,
                          int16_t,
                          uint8_t) {}
