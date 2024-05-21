// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "kernels/funcs/range_op.h"

namespace custom_kernel {

template <typename T, typename Context>
void FlashAttnUnpaddedMLUKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& q,
    const phi::DenseTensor& k,
    const phi::DenseTensor& v,
    const phi::DenseTensor& cu_seqlens_q,
    const phi::DenseTensor& cu_seqlens_k,
    const paddle::optional<phi::DenseTensor>& fixed_seed_offset,
    const paddle::optional<phi::DenseTensor>& attn_mask,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float scale,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    phi::DenseTensor* out,
    phi::DenseTensor* softmax,
    phi::DenseTensor* softmax_lse,
    phi::DenseTensor* seed_offset) {
  dev_ctx.template Alloc<T>(out);
  // q,k,v [total_*, num_heads, head_dim]
  auto dims = q.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));

  const int32_t total_q = dims[0];
  const int32_t num_heads = dims[1];
  const int32_t head_size = dims[2];

  const int32_t total_k = k.dims()[0];
  const int32_t num_heads_k = k.dims()[1];
  const int32_t batch_size = cu_seqlens_q.numel() - 1;

  PADDLE_ENFORCE_GT(
      batch_size,
      0,
      phi::errors::InvalidArgument(
          "flash_attn_raw receive input with batch_size should > 0"));

  PADDLE_ENFORCE_EQ(
      head_size % 8,
      0,
      phi::errors::InvalidArgument(
          "flash_attn_raw receive input with head_size should divisible by 8"));

  PADDLE_ENFORCE_LE(
      head_size,
      128,
      phi::errors::InvalidArgument(
          "flash_attn_raw receive input with head_size should <= 128"));

  PADDLE_ENFORCE_EQ(
      return_softmax,
      false,
      phi::errors::InvalidArgument(
          "flash_attn_raw receive input with return_softmax should be false"));

  phi::DenseTensor dropout_mask;
  void* dropout_mask_ptr = nullptr;
  if (return_softmax) {
    phi::DenseTensorMeta dropout_mask_meta = {
        phi::DataType::INT32, {num_heads, total_q, max_seqlen_k}};
    dropout_mask.set_meta(dropout_mask_meta);
    dropout_mask_ptr = dev_ctx.template Alloc<int32_t>(&dropout_mask);
  }

  // Generate random state for dropout and save for recompute in grad.
  uint64_t seed = 0;
  uint64_t offset = 0;
  if (fixed_seed_offset.get_ptr()) {
    const int64_t* fixed_seed_offset_data =
        fixed_seed_offset.get_ptr()->data<int64_t>();
    seed = static_cast<uint64_t>(fixed_seed_offset_data[0]);
    offset = static_cast<uint64_t>(fixed_seed_offset_data[1]);
  } else {
    seed = static_cast<uint64_t>(dev_ctx.GetGenerator()->Random64());
    offset = static_cast<uint64_t>(dev_ctx.GetGenerator()->Random64());
  }

  seed_offset->Resize({2});
  int64_t* seed_offset_data = dev_ctx.template HostAlloc<int64_t>(seed_offset);
  seed_offset_data[0] = seed;
  seed_offset_data[1] = offset;
  size_t* rng_state = reinterpret_cast<uint64_t*>(seed_offset_data);

  cnnlFlashAttentionDescriptor_t desc_;
  cnnlCreateFlashAttentionDescriptor(&desc_);
  auto compute_dtype = CNNL_DTYPE_FLOAT;
  auto prefer = CNNL_ACTIVATION_HIGH_PRECISION;
  auto attn_mask_mode = causal ? CNNL_ATTN_MASK_CAUSAL : CNNL_ATTN_MASK_NONE;
  cnnlSetFlashAttentionDescriptor(desc_,
                                  compute_dtype,
                                  prefer,
                                  attn_mask_mode,
                                  /*is_pack_mode = */ true,
                                  /*is_out_zero = */ false,
                                  return_softmax,
                                  max_seqlen_q,
                                  max_seqlen_k,
                                  dropout,
                                  scale);

  MLUCnnlTensorDesc query_desc(q, CNNL_LAYOUT_ARRAY, ToCnnlDataType(q.dtype()));
  MLUCnnlTensorDesc key_desc(k, CNNL_LAYOUT_ARRAY, ToCnnlDataType(q.dtype()));
  MLUCnnlTensorDesc value_desc(v, CNNL_LAYOUT_ARRAY, ToCnnlDataType(q.dtype()));

  MLUCnnlTensorDesc csq_desc(cu_seqlens_q);
  MLUCnnlTensorDesc csk_desc(cu_seqlens_k);

  softmax_lse->Resize({num_heads, total_q});
  dev_ctx.template Alloc<float>(softmax_lse);
  MLUCnnlTensorDesc softmax_lse_desc(*softmax_lse);

  out->Resize({total_q, num_heads, head_size});
  MLUCnnlTensorDesc out_desc(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(q.dtype()));

  // TODO(cifar10): Only equal-length input is supported. Variable lengths need
  // to be split and processed.
  MLUCnnl::FlashAttentionForward(dev_ctx,
                                 desc_,
                                 query_desc.get(),
                                 GetBasePtr(&q),
                                 key_desc.get(),
                                 GetBasePtr(&k),
                                 value_desc.get(),
                                 GetBasePtr(&v),
                                 csq_desc.get(),
                                 GetBasePtr(&cu_seqlens_q),
                                 csk_desc.get(),
                                 GetBasePtr(&cu_seqlens_k),
                                 rng_state,
                                 nullptr,  // dropout_mask_desc.desc()
                                 nullptr,  // dropout_mask_ptr
                                 softmax_lse_desc.get(),
                                 GetBasePtr(softmax_lse),
                                 out_desc.get(),
                                 GetBasePtr(out));
  out->Resize({batch_size, total_q / batch_size, num_heads, head_size});
}

template <typename T, typename Context>
void FlashAttnKernel(
    const Context& ctx,
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
  // q,k,v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  const int32_t batch_size = dims[0];
  const int32_t seqlen_q = dims[1];
  const int32_t num_heads = dims[2];
  const int32_t head_size = dims[3];
  const int32_t seqlen_k = k.dims()[1];
  const int32_t num_heads_k = k.dims()[2];
  const int32_t total_q = batch_size * seqlen_q;
  const int32_t total_k = batch_size * seqlen_k;
  const float scale = 1.0f / std::sqrt(head_size);

  phi::DenseTensor q_t_s, k_t_s, v_t_s;
  q_t_s = q;
  k_t_s = k;
  v_t_s = v;

  q_t_s.Resize({total_q, num_heads, head_size});
  k_t_s.Resize({total_k, num_heads, head_size});
  v_t_s.Resize({total_k, num_heads, head_size});

  phi::DenseTensor cu_seqlens_q;
  phi::DenseTensor cu_seqlens_k;
  ArangeRawKernel<int32_t, Context>(
      ctx, 0, (batch_size + 1) * seqlen_q, seqlen_q, &cu_seqlens_q);
  ArangeRawKernel<int32_t, Context>(
      ctx, 0, (batch_size + 1) * seqlen_k, seqlen_k, &cu_seqlens_k);

  FlashAttnUnpaddedMLUKernel<T, Context>(ctx,
                                         q_t_s,
                                         k_t_s,
                                         v_t_s,
                                         cu_seqlens_q,
                                         cu_seqlens_k,
                                         fixed_seed_offset,
                                         attn_mask,
                                         seqlen_q,
                                         seqlen_k,
                                         scale,
                                         dropout,
                                         causal,
                                         return_softmax,
                                         is_test,
                                         rng_name,
                                         out,
                                         softmax,
                                         softmax_lse,
                                         seed_offset);
}

template <typename T, typename Context>
void FlashAttnUnpaddedGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& q,
    const phi::DenseTensor& k,
    const phi::DenseTensor& v,
    const phi::DenseTensor& cu_seqlens_q,
    const phi::DenseTensor& cu_seqlens_k,
    const phi::DenseTensor& out,
    const phi::DenseTensor& softmax_lse,
    const phi::DenseTensor& seed_offset,
    const paddle::optional<phi::DenseTensor>& attn_mask,
    const phi::DenseTensor& dout,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float scale,
    float dropout,
    bool causal,
    phi::DenseTensor* dq,
    phi::DenseTensor* dk,
    phi::DenseTensor* dv) {
  dev_ctx.template Alloc<T>(dq);
  dev_ctx.template Alloc<T>(dk);
  dev_ctx.template Alloc<T>(dv);

  // q,k,v [total_*, num_heads, head_dim]
  auto dims = q.dims();
  const int32_t total_q = dims[0];
  const int32_t batch_size = cu_seqlens_q.numel() - 1;
  const int32_t num_heads = dims[1];
  const int32_t head_size = dims[2];
  const int32_t total_k = k.dims()[0];
  const int32_t num_heads_k = k.dims()[1];

  // const dout and out cannot be modified
  Tensor dout_tensor = dout;
  dout_tensor.Resize({total_q, num_heads, head_size});
  const int32_t head_size_og = dout_tensor.dims()[2];
  MLUCnnlTensorDesc diff_out_desc(dout_tensor);

  Tensor out_tensor = out;
  out_tensor.Resize({total_q, num_heads, head_size});
  MLUCnnlTensorDesc fwd_out_desc(
      out_tensor, CNNL_LAYOUT_ARRAY, ToCnnlDataType(q.dtype()));

  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      phi::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  PADDLE_ENFORCE_GT(
      batch_size,
      0,
      phi::errors::InvalidArgument(
          "flash_attn_raw receive input with batch_size should > 0"));

  PADDLE_ENFORCE_EQ(
      head_size % 8,
      0,
      phi::errors::InvalidArgument(
          "flash_attn_raw receive input head_size should divisible by 8"));

  PADDLE_ENFORCE_LE(head_size,
                    128,
                    phi::errors::InvalidArgument(
                        "flash_attn_raw receive input head_size should <=128"));

  cnnlFlashAttentionDescriptor_t desc_;
  cnnlCreateFlashAttentionDescriptor(&desc_);
  auto compute_dtype = CNNL_DTYPE_FLOAT;
  auto prefer = CNNL_ACTIVATION_HIGH_PRECISION;
  auto attn_mask_mode = causal ? CNNL_ATTN_MASK_CAUSAL : CNNL_ATTN_MASK_NONE;
  cnnlSetFlashAttentionBackwardDescriptor(desc_,
                                          compute_dtype,
                                          prefer,
                                          attn_mask_mode,
                                          /*is_pack_mode = */ true,
                                          /*is_out_zero = */ false,
                                          /*is_store_softmax_d = */ false,
                                          max_seqlen_q,
                                          max_seqlen_k,
                                          dropout,
                                          scale);

  MLUCnnlTensorDesc query_desc(q, CNNL_LAYOUT_ARRAY, ToCnnlDataType(q.dtype()));
  MLUCnnlTensorDesc key_desc(k, CNNL_LAYOUT_ARRAY, ToCnnlDataType(q.dtype()));
  MLUCnnlTensorDesc value_desc(v, CNNL_LAYOUT_ARRAY, ToCnnlDataType(q.dtype()));

  MLUCnnlTensorDesc csq_desc(cu_seqlens_q);
  MLUCnnlTensorDesc csk_desc(cu_seqlens_k);
  MLUCnnlTensorDesc softmax_lse_desc(softmax_lse);

  dq->Resize({total_q, num_heads, head_size});
  dk->Resize({total_k, num_heads, head_size});
  dv->Resize({total_k, num_heads, head_size});

  MLUCnnlTensorDesc diff_query_desc(
      *dq, CNNL_LAYOUT_ARRAY, ToCnnlDataType(q.dtype()));
  MLUCnnlTensorDesc diff_key_desc(
      *dk, CNNL_LAYOUT_ARRAY, ToCnnlDataType(q.dtype()));
  MLUCnnlTensorDesc diff_value_desc(
      *dv, CNNL_LAYOUT_ARRAY, ToCnnlDataType(q.dtype()));

  int64_t* seed_offset_data = const_cast<int64_t*>(seed_offset.data<int64_t>());
  size_t* rng_state = reinterpret_cast<uint64_t*>(seed_offset_data);

  MLUCnnl::FlashAttentionBackward(dev_ctx,
                                  desc_,
                                  diff_out_desc.get(),
                                  GetBasePtr(&dout_tensor),
                                  query_desc.get(),
                                  GetBasePtr(&q),
                                  key_desc.get(),
                                  GetBasePtr(&k),
                                  value_desc.get(),
                                  GetBasePtr(&v),
                                  fwd_out_desc.get(),
                                  GetBasePtr(&out_tensor),
                                  softmax_lse_desc.get(),
                                  GetBasePtr(&softmax_lse),
                                  csq_desc.get(),
                                  GetBasePtr(&cu_seqlens_q),
                                  csk_desc.get(),
                                  GetBasePtr(&cu_seqlens_k),
                                  rng_state,
                                  diff_query_desc.get(),
                                  GetBasePtr(dq),
                                  diff_key_desc.get(),
                                  GetBasePtr(dk),
                                  diff_value_desc.get(),
                                  GetBasePtr(dv));
  dq->Resize({batch_size, total_q / batch_size, num_heads, head_size});
  dk->Resize({batch_size, total_k / batch_size, num_heads, head_size});
  dv->Resize({batch_size, total_k / batch_size, num_heads, head_size});
}

template <typename T, typename Context>
void FlashAttnGradKernel(const Context& ctx,
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
  // q,k,v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();
  const int32_t batch_size = dims[0];
  const int32_t seqlen_q = dims[1];
  const int32_t num_heads = dims[2];
  const int32_t head_size_og = dout.dims()[3];
  const int32_t head_size = dims[3];
  const int32_t seqlen_k = k.dims()[1];
  const int32_t num_heads_k = k.dims()[2];

  const int32_t total_q = batch_size * seqlen_q;
  const int32_t total_k = batch_size * seqlen_k;

  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      phi::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  VLOG(10) << "FlashAttn bwd dims q[" << q.dims() << "], k[" << k.dims()
           << "], v[" << v.dims() << "]";

  const float scale = 1.0f / std::sqrt(head_size);
  phi::DenseTensor q_t_s, k_t_s, v_t_s;
  q_t_s = q;
  k_t_s = k;
  v_t_s = v;

  q_t_s.Resize({total_q, num_heads, head_size});
  k_t_s.Resize({total_k, num_heads, head_size});
  v_t_s.Resize({total_k, num_heads, head_size});

  phi::DenseTensor cu_seqlens_q;
  phi::DenseTensor cu_seqlens_k;
  ArangeRawKernel<int32_t, Context>(
      ctx, 0, (batch_size + 1) * seqlen_q, seqlen_q, &cu_seqlens_q);
  ArangeRawKernel<int32_t, Context>(
      ctx, 0, (batch_size + 1) * seqlen_k, seqlen_k, &cu_seqlens_k);

  FlashAttnUnpaddedGradKernel<T, Context>(ctx,
                                          q_t_s,
                                          k_t_s,
                                          v_t_s,
                                          cu_seqlens_q,
                                          cu_seqlens_k,
                                          out,
                                          softmax_lse,
                                          seed_offset,
                                          attn_mask,
                                          dout,
                                          seqlen_q,
                                          seqlen_k,
                                          scale,
                                          dropout,
                                          causal,
                                          dq,
                                          dk,
                                          dv);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(flash_attn_unpadded,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::FlashAttnUnpaddedMLUKernel,
                          phi::dtype::float16) {
  kernel->InputAt(5).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flash_attn,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::FlashAttnKernel,
                          phi::dtype::float16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flash_attn_unpadded_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::FlashAttnUnpaddedGradKernel,
                          phi::dtype::float16) {
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flash_attn_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::FlashAttnGradKernel,
                          phi::dtype::float16) {
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}
