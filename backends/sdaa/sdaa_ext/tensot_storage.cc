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

#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/extension.h"

std::vector<std::vector<int64_t>> CustomInferShape(
    const std::vector<int64_t>& x_shape) {
  return {{-1}};
}

std::vector<paddle::DataType> CustomInferDtype(
    const paddle::DataType& x_dtype) {
  return {paddle::DataType::INT64};
}

std::vector<paddle::Tensor> CustomTensorStorage(const paddle::Tensor& x) {
  auto* x_dense_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(x.impl()).get();
  int64_t storage_f = -1;
  if (x_dense_tensor->storage_properties_initialized()) {
    SDAAStorageProperties grad_properties =
        x_dense_tensor->storage_properties<SDAAStorageProperties>();
    storage_f = grad_properties.storage_format;
  }
  auto out = paddle::zeros({1}, paddle::DataType::INT64, phi::CPUPlace());
  auto* out_data = out.data<int64_t>();
  out_data[0] = storage_f;
  return {out};
}

PD_BUILD_OP(tensot_storage)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomTensorStorage))
    .SetInferShapeFn(PD_INFER_SHAPE(CustomInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(CustomInferDtype));
