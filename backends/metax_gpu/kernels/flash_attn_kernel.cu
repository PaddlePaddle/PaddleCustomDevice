// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
// Reserved. Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
// clang-format off
#include "flash_attn_kernel.h"  // NOLINT
#include "flash_attn_utils.h"   // NOLINT
#include "glog/logging.h"       // For VLOG()
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/api/ext/op_meta_info.h"
// clang-format on
namespace phi {

template <typename T, typename Context>
void FlashAttnUnpaddedKernel(
    const Context& ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const Scalar& max_seqlen_q_s,
    const Scalar& max_seqlen_k_s,
    float scale,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
#ifdef PADDLE_WITH_FLASHATTN
  ctx.template Alloc<T>(out);
  cudaStream_t stream = ctx.stream();

  int64_t max_seqlen_q = max_seqlen_q_s.to<int64_t>();
  int64_t max_seqlen_k = max_seqlen_k_s.to<int64_t>();

  // q, k, v [total_q/k/v, num_heads, head_dim]
  auto dims = q.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size = dims[2];
  const int64_t head_size_og = out->dims()[2];
  const int64_t num_heads_k = k.dims()[1];
  const int64_t total_q = dims[0];
  const int64_t total_k = k.dims()[0];
  FlashAttnParamsFwd<T> params = FlashAttnParamsFwd<T>(ctx,
                                                       attn_mask,
                                                       return_softmax,
                                                       *softmax,
                                                       q,
                                                       k,
                                                       v,
                                                       *out,
                                                       *softmax_lse,
                                                       is_test,
                                                       dropout,
                                                       causal,
                                                       fixed_seed_offset,
                                                       *seed_offset,
                                                       rng_name,
                                                       batch_size,
                                                       max_seqlen_q,
                                                       num_heads,
                                                       head_size,
                                                       max_seqlen_k,
                                                       num_heads_k);
  VLOG(10) << "FlashAttn fwd seed: " << params.seed_offset_data[0]
           << ", offset: " << params.seed_offset_data[1];
  auto flash_cu_seqlens_q = DenseTensorToMcFlashAttnTensor(cu_seqlens_q);
  auto flash_cu_seqlens_k = DenseTensorToMcFlashAttnTensor(cu_seqlens_k);
  mcflashattnStatus_t succ =
      phi::dynload::mha_varlen_fwd(params.batch_size,
                                   total_q,
                                   params.num_heads,
                                   total_k,
                                   params.num_heads_k,
                                   head_size_og,
                                   params.q,
                                   params.k,
                                   params.v,
                                   params.out,
                                   flash_cu_seqlens_q,
                                   flash_cu_seqlens_k,
                                   nullptr,
                                   params.alibi_slopes,
                                   params.softmax_lse,
                                   params.p,
                                   params.rng_state,
                                   params.seqlen_q,
                                   params.seqlen_k,
                                   params.p_dropout,
                                   params.softmax_scale,
                                   params.is_causal,
                                   params.window_size_left,
                                   params.window_size_right,
                                   params.stream,
                                   params.extend_parameter);
  phi::dynload::release_tensor(flash_cu_seqlens_q);
  phi::dynload::release_tensor(flash_cu_seqlens_k);
  CheckFlashAttnStatus(succ);
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnKernel(const Context& ctx,
                     const DenseTensor& q,
                     const DenseTensor& k,
                     const DenseTensor& v,
                     const paddle::optional<DenseTensor>& fixed_seed_offset,
                     const paddle::optional<DenseTensor>& attn_mask,
                     float dropout,
                     bool causal,
                     bool return_softmax,
                     bool is_test,
                     const std::string& rng_name,
                     DenseTensor* out,
                     DenseTensor* softmax,
                     DenseTensor* softmax_lse,
                     DenseTensor* seed_offset) {
#ifdef PADDLE_WITH_FLASHATTN
  ctx.template Alloc<T>(out);
  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];
  FlashAttnParamsFwd<T> params = FlashAttnParamsFwd<T>(ctx,
                                                       attn_mask,
                                                       return_softmax,
                                                       *softmax,
                                                       q,
                                                       k,
                                                       v,
                                                       *out,
                                                       *softmax_lse,
                                                       is_test,
                                                       dropout,
                                                       causal,
                                                       fixed_seed_offset,
                                                       *seed_offset,
                                                       rng_name,
                                                       batch_size,
                                                       seqlen_q,
                                                       num_heads,
                                                       head_size,
                                                       seqlen_k,
                                                       num_heads_k);

  VLOG(10) << "[FlashAttn Forward] q.shape=[" << q.dims() << "], k.shape=["
           << k.dims() << "], v.shape=[" << v.dims() << "]";
  VLOG(10) << "[FlashAttn Forward] dropout=" << dropout
           << ", seed=" << params.seed_offset_data[0]
           << ", offset=" << params.seed_offset_data[1];
  VLOG(10) << "[FlashAttn Forward] softmax_scale=" << params.softmax_scale;
  if (attn_mask.get_ptr()) {
    VLOG(10) << "[FlashAttn Forward] attn_mask.shape=["
             << (attn_mask.get_ptr())->dims() << "]";
  }

  mcflashattnStatus_t succ = phi::dynload::mha_fwd(params.batch_size,
                                                   params.seqlen_q,
                                                   params.num_heads,
                                                   params.seqlen_k,
                                                   params.num_heads_k,
                                                   params.head_size,
                                                   params.q,
                                                   params.k,
                                                   params.v,
                                                   params.out,
                                                   params.alibi_slopes,
                                                   params.attn_mask,
                                                   params.softmax_lse,
                                                   params.p,  // return softmax
                                                   params.rng_state,
                                                   params.p_dropout,
                                                   params.softmax_scale,
                                                   params.is_causal,
                                                   params.window_size_left,
                                                   params.window_size_right,
                                                   params.stream,
                                                   params.extend_parameter);
  CheckFlashAttnStatus(succ);
#else
  RaiseNotSupportedError();
#endif
}

const phi::CustomContext* getcontext(const paddle::Tensor& tensor) {
  return static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(tensor.place()));
}

phi::DenseTensor paddletensor2densortensor(const paddle::Tensor& paddletensor) {
  return *(static_cast<const phi::DenseTensor*>(paddletensor.impl().get()));
}

phi::DenseTensor* opt_paddletensor2densortensor_ptr(
    const paddle::optional<paddle::Tensor>& opt_paddletensor) {
  if (opt_paddletensor) {
    auto ptr = *(opt_paddletensor.get_ptr());
    return static_cast<phi::DenseTensor*>(ptr.impl().get());
  } else {
    return nullptr;
  }
}

std::vector<paddle::Tensor> FlashAttnKernelKVCache(
    const paddle::Tensor& q_,
    const paddle::Tensor& k_cache_,
    const paddle::Tensor& v_cache_,
    const paddle::optional<paddle::Tensor>& k_,
    const paddle::optional<paddle::Tensor>& v_,
    const paddle::Tensor& seqlens_k_,
    const paddle::optional<paddle::Tensor>& rotary_cos_,
    const paddle::optional<paddle::Tensor>& rotary_sin_,
    const paddle::optional<paddle::Tensor>& cache_batch_idx_,
    const paddle::optional<paddle::Tensor>& block_table_,
    bool causal,
    bool is_rotary_interleaved,
    int num_splits,
    float dropout,
    bool return_softmax) {
#ifdef PADDLE_WITH_FLASHATTN
  auto ctx = getcontext(q_);
  auto q = paddletensor2densortensor(q_);
  auto k_cache = paddletensor2densortensor(k_cache_);
  auto v_cache = paddletensor2densortensor(v_cache_);
  auto seqlens_k = paddletensor2densortensor(seqlens_k_);

  auto k = opt_paddletensor2densortensor_ptr(k_);
  auto v = opt_paddletensor2densortensor_ptr(v_);
  auto rotary_cos = opt_paddletensor2densortensor_ptr(rotary_cos_);
  auto rotary_sin = opt_paddletensor2densortensor_ptr(rotary_sin_);
  auto block_table = opt_paddletensor2densortensor_ptr(block_table_);
  auto cache_batch_idx = opt_paddletensor2densortensor_ptr(cache_batch_idx_);

  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn_KVCache receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size = dims[3];

  std::shared_ptr<phi::DenseTensor> out = std::make_shared<phi::DenseTensor>();
  std::vector<int64_t> out_dims = {batch_size, seqlen_q, num_heads, head_size};
  out->Resize(phi::make_ddim(out_dims));
  (*ctx).Alloc(out.get(), q.dtype());
  std::shared_ptr<phi::DenseTensor> softmax_lse =
      std::make_shared<phi::DenseTensor>();
  FlashAttnParamsFwdKVCache<float> params(*ctx,
                                          return_softmax,
                                          q,
                                          k_cache,
                                          v_cache,
                                          k,
                                          v,
                                          seqlens_k,
                                          rotary_cos,
                                          rotary_sin,
                                          cache_batch_idx,
                                          block_table,
                                          *out,
                                          *softmax_lse,
                                          causal,
                                          is_rotary_interleaved,
                                          num_splits,
                                          batch_size,
                                          seqlen_q,
                                          num_heads,
                                          head_size,
                                          dropout);
  mcflashattnStatus_t succ =
      phi::dynload::mha_fwd_kvcache(params.q,
                                    params.k_cache,
                                    params.v_cache,
                                    params.k,
                                    params.v,
                                    params.seqlens_k,
                                    params.rotary_cos,
                                    params.rotary_sin,
                                    params.cache_batch_idx,
                                    params.block_table,
                                    params.alibi_slopes,
                                    params.softmax_lse,
                                    params.out,
                                    params.softmax_scale,
                                    params.is_causal,
                                    params.window_size_left,
                                    params.window_size_right,
                                    params.is_rotary_interleaved,
                                    params.stream,
                                    params.num_splits,
                                    params.softmax_lse_accum,
                                    params.out_accum,
                                    params.extend_parameter);
  CheckFlashAttnStatus(succ);
  return {paddle::Tensor(out), paddle::Tensor(softmax_lse)};
#else
  RaiseNotSupportedError();
#endif
}

std::vector<std::vector<int64_t>> fusedattentionInferShape(
    std::vector<int64_t> q_shape,
    std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape) {
  return {q_shape, k_shape, v_shape};
}

}  // namespace phi

PD_REGISTER_PLUGIN_KERNEL(flash_attn_unpadded,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnUnpaddedKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flash_attn,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_BUILD_OP(flash_attn_kvcache)
    .Inputs({"q",
             "k_cache",
             "v_cache",
             paddle::Optional("k"),
             paddle::Optional("v"),
             "seqlens_k",
             paddle::Optional("rotary_cos"),
             paddle::Optional("rotary_sin"),
             paddle::Optional("cache_batch_idx"),
             paddle::Optional("block_table")})
    .Outputs({"out", "softmax_lse"})
    .Attrs({"causal:bool",
            "is_rotary_interleaved:bool",
            "num_splits:int",
            "dropout:float",
            "return_softmax:bool"})
    .SetKernelFn(PD_KERNEL(phi::FlashAttnKernelKVCache))
    .SetInferShapeFn(PD_INFER_SHAPE(phi::fusedattentionInferShape));
