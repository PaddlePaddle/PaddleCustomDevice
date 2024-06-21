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

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename Context>
void doAddNRaw(const Context& dev_ctx,
               const std::vector<const phi::DenseTensor*>& x,
               phi::DenseTensor* out) {
  std::vector<tecodnnTensorDescriptor_t> descs;
  std::vector<const void*> data_ptrs;
  // NOTE(liaotianju): addN enforce all tensor shapes are equal
  auto desc = sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(x[0]->dims()),
                                             x[0]->dtype(),
                                             TensorFormat::Undefined);
  for (size_t i = 0; i < x.size(); i++) {
    descs.push_back(desc);
    data_ptrs.push_back(x[i]->data());
  }
  phi::DenseTensor ptrs;
  int64_t ptr_size = x.size() * sizeof(void*);
  ptrs.Resize({ptr_size});
  dev_ctx.template Alloc<int8_t>(&ptrs);
  AsyncMemCpyH2D(nullptr,
                 static_cast<C_Stream>(dev_ctx.stream()),
                 ptrs.data(),
                 data_ptrs.data(),
                 ptr_size);
  auto out_desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(out->dims()), out->dtype(), TensorFormat::Undefined);
  auto handle = GetHandleFromCTX(dev_ctx);
  TECODNN_CHECK(tecodnnAddN(handle,
                            x.size(),
                            descs.data(),
                            reinterpret_cast<void**>(ptrs.data()),
                            out_desc,
                            out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(desc));
}

template <typename T, typename Context>
void AddNKernel(const Context& dev_ctx,
                const std::vector<const phi::DenseTensor*>& x,
                phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA AddNKernel";

  PADDLE_ENFORCE_EQ(out->dtype() == phi::DataType::FLOAT32 ||
                        out->dtype() == phi::DataType::FLOAT16,
                    true,
                    phi::errors::InvalidArgument(
                        "addN only support dytpe of float32 and float16"
                        "But recived: dtype is %s",
                        out->dtype()));
  if (UNLIKELY(x.size() == 0)) {
    return;
  }

  dev_ctx.template Alloc<T>(out);

  int n = static_cast<int>(x.size());
  if (n == 1) {
    phi::Copy(dev_ctx, *x[0], out->place(), false, out);
    return;
  }

  if (isEnvEnable("HIGH_PERFORMANCE_CONV")) {
    // x[0].storage_properties is true, which means grad comes from conv_filter
    // and is used in adam optimizer
    if (x[0]->storage_properties_initialized() && (x.size() != 2)) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "grad tensor's storage propertiey only supported in optimizer.Adam "
          "with weight_decay "));
    }

    // when add operator is performed in adam optimizer, grad tensor should be
    // the first input parameter.
    if ((!x[0]->storage_properties_initialized()) &&
        x[1]->storage_properties_initialized()) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "grad tensor's storage propertiey only supported in "
          "optimizer.Adam with weight_decay "));
    }

    if (x[0]->storage_properties_initialized()) {
      SDAAStorageProperties grad_properties;
      grad_properties = x[0]->storage_properties<SDAAStorageProperties>();
      // for amp case
      if (!x[1]->storage_properties_initialized()) {
        sdaa_ops::swapTensorData(dev_ctx, *x[1], grad_properties);
        sdaa_ops::doAddStorageProperties(dev_ctx, out, grad_properties);
        // for fp32 case
      } else {
        sdaa_ops::doAddStorageProperties(dev_ctx, out, grad_properties);
      }
    }
  }

  std::vector<phi::DenseTensor> inputs;
  std::vector<std::vector<int>> inputs_dims;
  for (int i = 0; i < n; i++) {
    if (x[i] && x[i]->numel() > 0) {
      inputs.push_back(*x[i]);
      std::vector<int> x_i_dims = phi::vectorize<int>(x[i]->dims());
      inputs_dims.push_back(x_i_dims);
    }
  }
  for (int i = 1; i < inputs_dims.size(); i++) {
    PADDLE_ENFORCE_EQ(inputs_dims[0],
                      inputs_dims[i],
                      phi::errors::InvalidArgument("input shape must be same"));
  }

  float alpha = 1.0;
  float beta = 0.0;
  int m = inputs.size();

  // special case for ppocr-cls
  if (m == 2) {
    sdaa_ops::doElementAdd(dev_ctx, inputs[0], inputs[1], -1, out);
    return;
  }

  doAddNRaw(dev_ctx, x, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add_n,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::AddNKernel,
                          float,
                          phi::dtype::float16) {}
