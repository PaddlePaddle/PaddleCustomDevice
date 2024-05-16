// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#if PADDLE_WITH_ATB

#include "fused_blha_layer_op_utils.h"  // NOLINT

std::vector<std::vector<int64_t>> RemovePaddingInferShape(
    const std::vector<int64_t>& x, const std::vector<int64_t>& seqlen) {
  std::vector<int64_t> out_dims{-1};  // ntokens
  return {out_dims};
}

std::vector<phi::DataType> RemovePaddingInferDType(
    const phi::DataType& x, const phi::DataType& seqlen) {
  return {x};
}

template <typename T>
void remove_padding(const phi::CustomContext& dev_ctx,
                    const void* in,
                    phi::DataType x_dtype,
                    const T* seqlen,
                    int64_t bsz,
                    int64_t padding_len,
                    void* out) {
  int64_t in_offset = 0, out_offset = 0;
  for (auto i = 0; i < bsz; ++i) {
    if (seqlen[i] > 0) {
      in_offset = i * padding_len * phi::SizeOf(x_dtype);
      ACL_CHECK(
          aclrtMemcpyAsync(out + out_offset,
                           seqlen[i] * phi::SizeOf(x_dtype),
                           in + in_offset,
                           seqlen[i] * phi::SizeOf(x_dtype),
                           ACL_MEMCPY_DEVICE_TO_DEVICE,
                           reinterpret_cast<aclrtStream>(dev_ctx.stream())));
      out_offset += seqlen[i] * phi::SizeOf(x_dtype);
    }
  }
}

std::vector<paddle::Tensor> RemovePaddingOp(const paddle::Tensor& x,
                                            const paddle::Tensor& seqlen) {
  auto place = x.place();
  const auto& dev_ctx = *static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(place));

  auto x_shape = x.shape();
  const int bsz = x_shape[0];
  const int padding_len = x_shape[1];

  auto seqlen_host = seqlen.copy_to(paddle::CPUPlace(), true);
  int64_t token_num = 0;
  if (seqlen.dtype() == phi::DataType::INT32) {
    auto* seqlen_data = seqlen_host.data<int32_t>();
    token_num =
        std::accumulate(seqlen_data, seqlen_data + seqlen_host.numel(), 0);
  } else {
    auto* seqlen_data = seqlen_host.data<int64_t>();
    token_num =
        std::accumulate(seqlen_data, seqlen_data + seqlen_host.numel(), 0);
  }

  paddle::Tensor out(x.place());
  init_tensor(dev_ctx, x.dtype(), {token_num}, &out);
  if (seqlen.dtype() == phi::DataType::INT32) {
    remove_padding(dev_ctx,
                   x.data(),
                   x.dtype(),
                   seqlen_host.data<int32_t>(),
                   bsz,
                   padding_len,
                   out.data());
  } else {
    remove_padding(dev_ctx,
                   x.data(),
                   x.dtype(),
                   seqlen_host.data<int64_t>(),
                   bsz,
                   padding_len,
                   out.data());
  }
  return {out};
}

PD_BUILD_OP(remove_padding)
    .Inputs({"x", "seqlen"})
    .Outputs({"x_remove_padding"})
    .Attrs({})
    .SetKernelFn(PD_KERNEL(RemovePaddingOp))
    .SetInferShapeFn(PD_INFER_SHAPE(RemovePaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RemovePaddingInferDType));
#endif
