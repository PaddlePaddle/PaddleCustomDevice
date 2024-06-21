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

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
#include "sdcops.h"  //NOLINT

namespace custom_kernel {

struct TensorStride {
  uint32_t lda;
  uint32_t stride;
};

void CheckInputs(const phi::DenseTensor& q,
                 const phi::DenseTensor& k,
                 const phi::DenseTensor& v,
                 float dropout) {
  // q,k,v [seq_len, batch_size, num_heads, head_dim]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[seq_len, batch_size, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      q.dtype() == k.dtype(),
      true,
      phi::errors::InvalidArgument("flash_attn q k v dtype must be the same"
                                   "but receive q:{%d} k:{%d}",
                                   q.dtype(),
                                   k.dtype()));
  PADDLE_ENFORCE_EQ(
      q.dtype() == v.dtype(),
      true,
      phi::errors::InvalidArgument("flash_attn q k v dtype must be the same"
                                   "but receive q:{%d} v:{%d}",
                                   q.dtype(),
                                   v.dtype()));
  PADDLE_ENFORCE_EQ(
      dropout <= 0.0,
      true,
      phi::errors::InvalidArgument("flash_attn not support dropout yet"
                                   "but receive dropout:{%f}",
                                   dropout));
}

void CastFP32TOFP16Raw(const Context& dev_ctx,
                       const phi::DenseTensor& src,
                       void* dst) {
  std::vector<int> src_dims(phi::vectorize<int>(src.dims()));
  tecodnnTensorDescriptor_t src_Desc =
      sdaa_ops::GetTecodnnTensorDesc(src_dims, src.dtype(), TensorFormat::NCHW);
  tecodnnTensorDescriptor_t dst_Desc = sdaa_ops::GetTecodnnTensorDesc(
      src_dims, phi::DataType::FLOAT16, TensorFormat::NCHW);
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  float alpha = 1.0, beta = 0.0;
  TECODNN_CHECK(tecodnnTransformTensor(
      tecodnnHandle, &alpha, src_Desc, src.data(), &beta, dst_Desc, dst));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(src_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dst_Desc));
}

int64_t GetFP16TensorSize(const phi::DenseTensor& t) {
  return phi::SizeOf(phi::DataType::FLOAT16) * t.numel();
}

TensorStride GenTensorStride(const phi::DenseTensor& t) {
  // t [seq_len, batch_size, num_heads, head_dim]
  auto dims = t.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      4,
      phi::errors::InvalidArgument("stride calculate only support 4D"));
  TensorStride t_stride;
  t_stride.lda = dims[1] * dims[2] * dims[3];
  t_stride.stride = dims[3];
  return t_stride;
}

template <typename T, typename Context>
void FlashAttnKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& q,
    const phi::DenseTensor& k,
    const phi::DenseTensor& v,
    const paddle::optional<phi::DenseTensor>& fixed_seed_offset,
    const paddle::optional<phi::DenseTensor>& attn_mask,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    phi::DenseTensor* out,
    phi::DenseTensor* softmax,
    phi::DenseTensor* softmax_lse,
    phi::DenseTensor* seed_offset) {
  VLOG(4) << "Call SDAA FlashAttnKernel";
  // q,k,v [seq_len, batch_size, num_heads, head_dim]
  CheckInputs(q, k, v, dropout);
  // prepare param
  auto q_dims = q.dims();
  uint32_t attn_seq_len = q_dims[0];
  uint32_t attn_size_per_head = q_dims[3];
  uint32_t attn_head_num = q_dims[1] * q_dims[2];
  const float qk_scalar = 1.0f / std::sqrt(attn_size_per_head);
  FLASH_ATTENTION_MODE attn_mode = is_test ? INFER_MODE : TRAIN_MODE;
  // gen workspace only train mode
  int64_t workspace_size = 0;
  int64_t softmax_offset = 0;
  if (attn_mode == TRAIN_MODE) {
    workspace_size = lmik::flash_attention_get_size(
        attn_seq_len, attn_head_num, attn_size_per_head, attn_mode);
    softmax_offset = workspace_size;
  }
  // get qkv size
  int64_t half_q_size = GetFP16TensorSize(q);
  int64_t half_k_size = GetFP16TensorSize(k);
  int64_t half_v_size = GetFP16TensorSize(v);
  workspace_size += half_q_size + half_k_size + half_v_size;
  VLOG(4) << "workspace_size:" << workspace_size;
  softmax_lse->Resize({workspace_size});
  dev_ctx.Alloc(softmax_lse, DataType::INT8);
  std::uintptr_t softmax_addr =
      reinterpret_cast<uintptr_t>(softmax_lse->data());
  void* half_q_addr = reinterpret_cast<void*>(softmax_addr + softmax_offset);
  void* half_k_addr =
      reinterpret_cast<void*>(softmax_addr + softmax_offset + half_q_size);
  void* half_v_addr = reinterpret_cast<void*>(softmax_addr + softmax_offset +
                                              half_q_size + half_k_size);
  CastFP32TOFP16Raw(dev_ctx, q, half_q_addr);
  CastFP32TOFP16Raw(dev_ctx, k, half_k_addr);
  CastFP32TOFP16Raw(dev_ctx, v, half_v_addr);

  VLOG(4) << "scale:" << qk_scalar;
  // calculate ld && stride
  TensorStride q_stride = GenTensorStride(q);
  TensorStride k_stride = GenTensorStride(k);
  TensorStride v_stride = GenTensorStride(v);

  // float out
  dev_ctx.template Alloc<T>(out);
  TensorStride out_stride = GenTensorStride(*out);
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);
  TCUS_CHECK(
      lmik::flash_attention<float>(half_q_addr,
                                   q_stride.lda,
                                   q_stride.stride,
                                   half_k_addr,
                                   k_stride.lda,
                                   k_stride.stride,
                                   half_v_addr,
                                   v_stride.lda,
                                   v_stride.stride,
                                   attn_seq_len,
                                   attn_size_per_head,
                                   attn_head_num,
                                   out->data<T>(),
                                   out_stride.lda,
                                   out_stride.stride,
                                   qk_scalar,
                                   attn_mode,
                                   static_cast<float*>(softmax_lse->data()),
                                   custom_stream));
}

template <typename T, typename Context>
void FlashAttnGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& q,
                         const phi::DenseTensor& k,
                         const phi::DenseTensor& v,
                         const phi::DenseTensor& out,
                         const phi::DenseTensor& softmax_lse,
                         const phi::DenseTensor& seed_offset,
                         const paddle::optional<phi::DenseTensor>& attn_mask,
                         const phi::DenseTensor& dout,
                         float dropout,
                         bool causal,
                         phi::DenseTensor* dq,
                         phi::DenseTensor* dk,
                         phi::DenseTensor* dv) {
  VLOG(4) << "Call SDAA FlashAttnGradKernel";
  // q,k,v [seq_len, batch_size, num_heads, head_dim]
  CheckInputs(q, k, v, dropout);
  // prepare dq,dk,dv
  dev_ctx.template Alloc<T>(dq);
  dev_ctx.template Alloc<T>(dk);
  dev_ctx.template Alloc<T>(dv);

  // prepare param
  FLASH_ATTENTION_MODE attn_mode = TRAIN_MODE;
  TensorStride q_stride = GenTensorStride(q);
  TensorStride k_stride = GenTensorStride(k);
  TensorStride v_stride = GenTensorStride(v);
  TensorStride dout_stride = GenTensorStride(dout);

  auto q_dims{q.dims()};
  uint32_t attn_seq_len = q_dims[0];
  uint32_t attn_size_per_head = q_dims[3];
  uint32_t attn_head_num = q_dims[1] * q_dims[2];
  const float qk_scalar = 1.0f / std::sqrt(attn_size_per_head);
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);
  bool use_float_grad = true;

  // get qkv addr
  int64_t softmax_offset = lmik::flash_attention_get_size(
      attn_seq_len, attn_head_num, attn_size_per_head, attn_mode);
  int64_t half_q_size = GetFP16TensorSize(q);
  int64_t half_k_size = GetFP16TensorSize(k);
  int64_t half_v_size = GetFP16TensorSize(v);
  std::uintptr_t softmax_addr = reinterpret_cast<uintptr_t>(softmax_lse.data());
  void* half_q_addr = reinterpret_cast<void*>(softmax_addr + softmax_offset);
  void* half_k_addr =
      reinterpret_cast<void*>(softmax_addr + softmax_offset + half_q_size);
  void* half_v_addr = reinterpret_cast<void*>(softmax_addr + softmax_offset +
                                              half_q_size + half_k_size);

  // launch kernel
  TCUS_CHECK(
      lmik::flash_attention_bkd<float>(half_q_addr,
                                       q_stride.lda,
                                       q_stride.stride,
                                       half_k_addr,
                                       k_stride.lda,
                                       k_stride.stride,
                                       half_v_addr,
                                       v_stride.lda,
                                       v_stride.stride,
                                       dq->data(),
                                       q_stride.lda,
                                       q_stride.stride,
                                       dk->data(),
                                       k_stride.lda,
                                       k_stride.stride,
                                       dv->data(),
                                       v_stride.lda,
                                       v_stride.stride,
                                       use_float_grad,
                                       attn_seq_len,
                                       attn_size_per_head,
                                       attn_head_num,
                                       const_cast<float*>(dout.data<T>()),
                                       dout_stride.lda,
                                       dout_stride.stride,
                                       qk_scalar,
                                       1.0f,
                                       const_cast<void*>(softmax_lse.data()),
                                       attn_mode,
                                       custom_stream));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    flash_attn, sdaa, ALL_LAYOUT, custom_kernel::FlashAttnKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(flash_attn_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::FlashAttnGradKernel,
                          float) {}
