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
void LabelSmoothKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const paddle::optional<phi::DenseTensor>& dist,
                       float epsilon,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto epsilon_gt = 1.0f - epsilon;
  if (x.numel() == 0) return;
  auto label_dim = x.dims()[x.dims().size() - 1];

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  auto data_type = ToCnnlDataType<T>();
  MLUCnnlOpTensorDesc op_tensor_desc(
      CNNL_OP_TENSOR_ADD, data_type, CNNL_NOT_PROPAGATE_NAN);
  if (dist) {
    const auto* dist_ptr = dist.get_ptr();
    MLUCnnlTensorDesc dist_desc(*dist_ptr);
    MLUCnnl::OpTensor(dev_ctx,
                      op_tensor_desc.get(),
                      x_desc.get(),
                      GetBasePtr(&x),
                      dist_desc.get(),
                      GetBasePtr(dist_ptr),
                      out_desc.get(),
                      GetBasePtr(out),
                      data_type,
                      epsilon_gt,
                      epsilon);
  } else {
    Tensor dist_tensor;
    dist_tensor.Resize({1, label_dim});
    dev_ctx.template Alloc<T>(&dist_tensor);
    MLUCnnlTensorDesc dist_desc(dist_tensor);
    auto value = static_cast<T>(1.0f / label_dim);
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &value,
                  dist_desc.get(),
                  GetBasePtr(&dist_tensor));
    MLUCnnl::OpTensor(dev_ctx,
                      op_tensor_desc.get(),
                      x_desc.get(),
                      GetBasePtr(&x),
                      dist_desc.get(),
                      GetBasePtr(&dist_tensor),
                      out_desc.get(),
                      GetBasePtr(out),
                      data_type,
                      epsilon_gt,
                      epsilon);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(label_smooth,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::LabelSmoothKernel,
                          float,
                          phi::dtype::float16) {}
