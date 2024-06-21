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

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "kernels/profiler/sdaa_wrapper.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/all_context.h"
#include "sdcops.h"  // NOLINT

#define CHECK_CUSTOM_INPUT(x) \
  PD_CHECK(x.is_custom_device(), #x " must be a custom Tensor.")

// ONLY use x and y shape
std::vector<std::vector<int64_t>> RopeInferShape(
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& k_shape,
    const std::vector<int64_t>& cos_shape,
    const std::vector<int64_t>& sin_shape) {
  int bs, seq_len, head_nums, size_per_head;
  seq_len = q_shape[0];
  bs = q_shape[1];
  head_nums = q_shape[2];
  size_per_head = q_shape[3];
  std::vector<int64_t> q_out_shape = {seq_len, bs, head_nums, size_per_head};
  std::vector<int64_t> k_out_shape = {seq_len, bs, head_nums, size_per_head};
  return {q_out_shape, k_out_shape};
}

std::vector<paddle::DataType> RopeInferDtype(const paddle::DataType& q_dtype,
                                             const paddle::DataType& k_dtype) {
  return {q_dtype, k_dtype};
}

std::vector<paddle::Tensor> CustomRopeForward(const paddle::Tensor& q,
                                              const paddle::Tensor& k,
                                              const paddle::Tensor& cos,
                                              const paddle::Tensor& sin) {
  // check imput and get custom device context
  VLOG(4) << "check imput and get custom device context...";
  CHECK_CUSTOM_INPUT(q);
  CHECK_CUSTOM_INPUT(k);
  CHECK_CUSTOM_INPUT(cos);
  CHECK_CUSTOM_INPUT(sin);

  auto dev_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(q.place());
  auto custom_ctx = static_cast<const phi::CustomContext*>(dev_ctx);
  sdaaStream_t custom_stream = custom_kernel::GetStreamFromCTX(*custom_ctx);

  // check shape, dtype
  PADDLE_ENFORCE_EQ(q.dims().size() == 4 && k.dims().size() == 4 &&
                        cos.dims().size() == 4 && sin.dims().size() == 4,
                    true,
                    phi::errors::InvalidArgument(
                        "rotary_embedding inputs only support Tensor4D."));
  PADDLE_ENFORCE_EQ(
      q.dtype() == phi::DataType::FLOAT32 &&
          k.dtype() == phi::DataType::FLOAT32 &&
          cos.dtype() == phi::DataType::FLOAT32 &&
          sin.dtype() == phi::DataType::FLOAT32,
      true,
      phi::errors::InvalidArgument("rotary_embedding dtype should be float."));
  PADDLE_ENFORCE_EQ(q.dims()[0] == k.dims()[0] && q.dims()[1] == k.dims()[1] &&
                        q.dims()[2] == k.dims()[2] &&
                        q.dims()[3] == k.dims()[3],
                    true,
                    phi::errors::InvalidArgument(
                        "rotary_embedding inputs (q, k) shape must be equel."));
  PADDLE_ENFORCE_EQ(cos.dims()[2] == 1 && sin.dims()[2] == 1,
                    true,
                    phi::errors::InvalidArgument(
                        "rotary_embedding cos and sin shape[2] must equal 1."));
  PADDLE_ENFORCE_EQ(cos.dims()[3] == 128 && sin.dims()[3] == 128 &&
                        q.dims()[3] == 128 && k.dims()[3] == 128,
                    true,
                    phi::errors::InvalidArgument(
                        "rotary_embedding inputs shape[3] must be 128."));

  // get seq_len, head_nums, size_per_head
  int bs, seq_len, head_nums, size_per_head;
  seq_len = q.dims()[0];
  bs = q.dims()[1];
  head_nums = q.dims()[2];
  size_per_head = q.dims()[3];
  VLOG(4) << "get batch_size:" << bs << ", seq_len:" << seq_len
          << ", head_nums:" << head_nums << ", size_per_head:" << size_per_head;

  auto q_out = paddle::empty({seq_len, bs, head_nums, size_per_head},
                             phi::DataType::FLOAT32,
                             q.place());
  auto k_out = paddle::empty({seq_len, bs, head_nums, size_per_head},
                             phi::DataType::FLOAT32,
                             k.place());

  VLOG(4) << "CustomRopeForward";
  TCUS_CHECK(sdcops::rotary_embedding_ext(q.data<float>(),
                                          bs * head_nums * size_per_head,
                                          size_per_head,
                                          k.data<float>(),
                                          bs * head_nums * size_per_head,
                                          size_per_head,
                                          sin.data<float>(),
                                          cos.data<float>(),
                                          q_out.data<float>(),
                                          bs * head_nums * size_per_head,
                                          size_per_head,
                                          k_out.data<float>(),
                                          bs * head_nums * size_per_head,
                                          size_per_head,
                                          seq_len,
                                          head_nums * bs,
                                          head_nums * bs,
                                          size_per_head,
                                          size_per_head,
                                          true,
                                          true,
                                          DATA_FLOAT,
                                          DATA_FLOAT,
                                          custom_stream));

  return {q_out, k_out};
}

std::vector<paddle::Tensor> CustomRopeBackward(
    const paddle::Tensor& q,
    const paddle::Tensor& k,
    const paddle::Tensor& cos,
    const paddle::Tensor& sin,
    const paddle::Tensor& grad_q_out,
    const paddle::Tensor& grad_k_out) {
  VLOG(4) << "call CustomRopeBackward";

  // check imput and get custom device context
  CHECK_CUSTOM_INPUT(q);
  CHECK_CUSTOM_INPUT(k);
  CHECK_CUSTOM_INPUT(cos);
  CHECK_CUSTOM_INPUT(sin);
  CHECK_CUSTOM_INPUT(grad_q_out);
  CHECK_CUSTOM_INPUT(grad_k_out);

  // get ctx and stream
  auto dev_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(q.place());
  auto custom_ctx = static_cast<const phi::CustomContext*>(dev_ctx);
  sdaaStream_t custom_stream = custom_kernel::GetStreamFromCTX(*custom_ctx);

  // get seq_len, head_nums, size_per_head
  int bs, seq_len, head_nums, size_per_head;
  seq_len = q.dims()[0];
  bs = q.dims()[1];
  head_nums = q.dims()[2];
  size_per_head = q.dims()[3];

  auto grad_q = paddle::empty({seq_len, bs, head_nums, size_per_head},
                              phi::DataType::FLOAT32,
                              q.place());
  auto grad_k = paddle::empty({seq_len, bs, head_nums, size_per_head},
                              phi::DataType::FLOAT32,
                              k.place());

  VLOG(4) << "CustomRopeBackward";
  TCUS_CHECK(
      sdcops::rotary_embedding_backward_ext(grad_q_out.data<float>(),
                                            bs * head_nums * size_per_head,
                                            size_per_head,
                                            grad_k_out.data<float>(),
                                            bs * head_nums * size_per_head,
                                            size_per_head,
                                            sin.data<float>(),
                                            cos.data<float>(),
                                            grad_q.data<float>(),
                                            bs * head_nums * size_per_head,
                                            size_per_head,
                                            grad_k.data<float>(),
                                            bs * head_nums * size_per_head,
                                            size_per_head,
                                            seq_len,
                                            head_nums * bs,
                                            head_nums * bs,
                                            size_per_head,
                                            size_per_head,
                                            true,
                                            true,
                                            DATA_FLOAT,
                                            DATA_FLOAT,
                                            custom_stream));

  return {grad_q, grad_k};
}

PD_BUILD_OP(custom_fused_rotary_position_embedding)
    .Inputs({"q", "k", "cos", "sin"})
    .Outputs({"q_out", "k_out"})
    .SetKernelFn(PD_KERNEL(CustomRopeForward))
    .SetInferShapeFn(PD_INFER_SHAPE(RopeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RopeInferDtype));

PD_BUILD_GRAD_OP(custom_fused_rotary_position_embedding)
    .Inputs(
        {"q", "k", "cos", "sin", paddle::Grad("q_out"), paddle::Grad("k_out")})
    .Outputs({paddle::Grad("q"), paddle::Grad("k")})
    .SetKernelFn(PD_KERNEL(CustomRopeBackward));
