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

#include <algorithm>
#include <vector>

#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/all_context.h"
#include "sdcops.h"  // NOLINT

#define CHECK_CUSTOM_INPUT(x) \
  PD_CHECK(x.is_custom_device(), #x " must be a custom Tensor.")

void* GetStream(const paddle::Tensor& x) {
  CHECK_CUSTOM_INPUT(x);

  auto dev_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(x.place());
  auto custom_ctx = static_cast<const phi::CustomContext*>(dev_ctx);
  void* stream = custom_ctx->stream();
  PD_CHECK(stream != nullptr);

  return stream;
}

std::vector<std::vector<int64_t>> AddInferShape(
    const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape) {
  size_t s1 = x_shape.size(), s2 = y_shape.size();
  auto n = std::max(s1, s2);
  std::vector<int64_t> out_shape(n, 1);
  for (size_t i = 0; i < n; ++i) {
    if (i >= n - s1) {
      out_shape[i] = std::max(out_shape[i], x_shape[i + n - s1]);
    }
    if (i >= n - s2) {
      out_shape[i] = std::max(out_shape[i], y_shape[i + n - s2]);
    }
  }
  return {out_shape};
}

std::vector<paddle::DataType> AddInferDtype(const paddle::DataType& x_dtype,
                                            const paddle::DataType& y_dtype) {
  return {x_dtype};
}

std::vector<paddle::Tensor> AddForward(const paddle::Tensor& x,
                                       const paddle::Tensor& y) {
  CHECK_CUSTOM_INPUT(x);
  auto dev_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(x.place());
  auto custom_ctx = static_cast<const phi::CustomContext*>(dev_ctx);

  auto out_shape = AddInferShape(x.shape(), y.shape())[0];
  auto out = paddle::empty(out_shape, x.dtype(), x.place());
  auto workspace = paddle::empty(
      {10000 * out_shape.size()}, phi::DataType::INT64, x.place());
  int n = out_shape.size();
  std::vector<int64_t> x_expand(n, 1), y_expand(n, 1);
  for (int i = 0; i < x.shape().size(); ++i) {
    x_expand[i + n - x.shape().size()] = x.shape()[i];
  }
  for (int i = 0; i < y.shape().size(); ++i) {
    y_expand[i + n - y.shape().size()] = y.shape()[i];
  }
  sdaaStream_t custom_stream = custom_kernel::GetStreamFromCTX(*custom_ctx);

  float scale = 1.0;
  sdcops::binary_ops_tt(x.data(),
                        x_expand.data(),
                        x_expand.size(),
                        y.data(),
                        y_expand.data(),
                        y_expand.size(),
                        out.data(),
                        out_shape.data(),
                        out_shape.size(),
                        &scale,
                        BINARY_ADD,
                        DATA_FLOAT,
                        workspace.data(),
                        custom_stream);
  return {out};
}

PD_BUILD_OP(custom_add)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AddForward))
    .SetInferShapeFn(PD_INFER_SHAPE(AddInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AddInferDtype));
