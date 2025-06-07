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

#include <cstddef>

#include "./flash_attn_utils.h"
#include "glog/logging.h"  // For VLOG()
#include "paddle/common/flags.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/pad_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"
COMMON_DECLARE_int32(ixdnn_imp_mode);

namespace phi {
template <typename OutT>
struct ZeroFunctor {
  __device__ __forceinline__ OutT operator()() const {
    return static_cast<OutT>(0);
  }
};

template <typename T, typename Context>
void FlashAttnUnpaddedBaseKernel(
    const Context& ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const Scalar& max_seqlen_q_,
    const Scalar& max_seqlen_k_,
    float scale,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset,
    bool varlen_padded) {
  if (!out->IsInitialized()) ctx.template Alloc<T>(out);
  if (varlen_padded) {
    std::vector<const DenseTensor*> inputs{};
    std::vector<DenseTensor*> outputs{out};

    phi::funcs::ElementwiseKernel<T>(ctx, inputs, &outputs, ZeroFunctor<T>());
  }
  cudaStream_t stream = ctx.stream();

  // q, k, v [total_q/k/v, num_heads, head_dim]
  auto dims = q.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      k.dims().size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      v.dims().size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      out->dims().size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size = dims[2];
  const int64_t num_heads_k = k.dims()[1];
  int64_t max_seqlen_q = max_seqlen_q_.to<int64_t>();
  int64_t max_seqlen_k = max_seqlen_k_.to<int64_t>();

  // TODO(umiswing): add shape check
  // ixdnn
  int64_t total_q = dims[0];
  bool is_unpad = (total_q == batch_size * max_seqlen_q) ? false : true;
  const int64_t head_size_rounded = head_size + 32 - head_size % 32;

  DenseTensor q_padded, k_padded, v_padded;
  q_padded = q;
  k_padded = k;
  v_padded = v;

  if (is_unpad) {
    softmax_lse->Resize({num_heads, total_q});
  } else {
    softmax_lse->Resize({batch_size, num_heads, max_seqlen_q});
  }
  ctx.template Alloc<float>(softmax_lse);

  if (return_softmax) {
    softmax->Resize({batch_size, num_heads, max_seqlen_q, max_seqlen_k});
    ctx.template Alloc<T>(softmax);
  }

  int64_t seed;
  int64_t offset;

  if (fixed_seed_offset.get_ptr()) {
    const int64_t* fixed_seed_offset_data =
        fixed_seed_offset.get_ptr()->data<int64_t>();
    seed = static_cast<int64_t>(fixed_seed_offset_data[0]);
    offset = static_cast<int64_t>(fixed_seed_offset_data[1]);
  } else {
    uint64_t inc = batch_size * num_heads * 64;
    std::pair<int64_t, int64_t> seed_offset_pair;
    if (rng_name != "") {
      auto gen = phi::GetRandomSeedGenerator(rng_name);
      seed_offset_pair = gen->IncrementOffset(inc);
    } else {
      auto* gen = ctx.GetGenerator();
      seed_offset_pair = gen->IncrementOffset(inc);
    }
    seed = seed_offset_pair.first;
    offset = seed_offset_pair.second;
  }
  PhiloxCudaState philox_state = PhiloxCudaState(seed, offset);

  seed_offset->Resize({2});
  int64_t* seed_offset_data = ctx.template HostAlloc<int64_t>(seed_offset);
  seed_offset_data[0] = static_cast<int64_t>(seed);
  seed_offset_data[1] = static_cast<int64_t>(offset);

  cudnnFlashAttnConfigInfo flashAttnInfo;
  flashAttnInfo.softmax_scale = std::sqrt(1.f / head_size);
  flashAttnInfo.dropout_prob = is_test ? 0.0f : dropout;
  flashAttnInfo.is_causal = causal;
  // flashAttnInfo.is_alibi              = use_alibi;
  // flashAttnInfo.alibi_mode            = alibi_mode;
  flashAttnInfo.return_softmax_lse = true;
  flashAttnInfo.philox_args =
      *(reinterpret_cast<cudnnPhiloxCudaState*>(&philox_state));
  flashAttnInfo.imp_mode = FLAGS_ixdnn_imp_mode ? CUDNN_FATTN_LEAST_MEM_MODE
                                                : CUDNN_FATTN_BALANCE_MODE;
  flashAttnInfo.is_unpad = is_unpad;
  flashAttnInfo.batch = batch_size;
  flashAttnInfo.max_seq_len_src = max_seqlen_q;
  flashAttnInfo.max_seq_len_trg = max_seqlen_k;

  int32_t nb_dims = is_unpad ? 3 : 4;
  std::vector<int32_t> qShape, kShape, vShape, oShape, lseShape;
  std::vector<int32_t> qStride, kStride, vStride, oStride, lseStride;

  cudnnDataType_t dataType;
  if (q.dtype() == phi::DataType::FLOAT16) {
    dataType = CUDNN_DATA_HALF;
  } else if (q.dtype() == phi::DataType::BFLOAT16) {
    dataType = CUDNN_DATA_BFLOAT16;
  }

  cudnnHandle_t cudnn;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnCreate(&cudnn));

  cudnnFlashAttnDescriptor_t flashAttnDesc;
  cudnnTensorDescriptor_t q_desc;
  cudnnTensorDescriptor_t k_desc;
  cudnnTensorDescriptor_t v_desc;
  cudnnTensorDescriptor_t o_desc;
  cudnnTensorDescriptor_t lse_desc;

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateFlashAttnDescriptor(&flashAttnDesc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&q_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&k_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&v_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&o_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&lse_desc));

  DenseTensor q_, k_, v_;

  if (!is_unpad) {
    q_.Resize({batch_size, max_seqlen_q, num_heads, q_padded.dims()[2]});
    k_.Resize({batch_size, max_seqlen_k, num_heads_k, k_padded.dims()[2]});
    v_.Resize({batch_size, max_seqlen_k, num_heads_k, v_padded.dims()[2]});
    out->Resize(q_.dims());
  } else {
    q_ = q_padded;
    k_ = k_padded;
    v_ = v_padded;
  }

  copyDimsAndStrides(q_, qShape, qStride);
  copyDimsAndStrides(k_, kShape, kStride);
  copyDimsAndStrides(v_, vShape, vStride);
  copyDimsAndStrides(out, oShape, oStride);
  copyDimsAndStrides(softmax_lse, lseShape, lseStride);

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      q_desc, dataType, nb_dims, qShape.data(), qStride.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      k_desc, dataType, nb_dims, kShape.data(), kStride.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      v_desc, dataType, nb_dims, vShape.data(), vStride.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      o_desc, dataType, nb_dims, oShape.data(), oStride.data()));
  if (is_unpad) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnSetTensorNdDescriptor_lowerbound_2(
            lse_desc,
            CUDNN_DATA_FLOAT,
            nb_dims - 1,
            lseShape.data(),
            lseStride.data()));
  } else {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnSetTensorNdDescriptor(lse_desc,
                                                 CUDNN_DATA_FLOAT,
                                                 nb_dims - 1,
                                                 lseShape.data(),
                                                 lseStride.data()));
  }

  size_t size_tmpbuf = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnGetFlashAttnBuffers(cudnn,
                                             flashAttnDesc,
                                             batch_size,
                                             num_heads,
                                             max_seqlen_q,
                                             max_seqlen_k,
                                             head_size,
                                             total_q,
                                             is_unpad,
                                             false,
                                             &size_tmpbuf,
                                             flashAttnInfo.imp_mode));

  auto workspace_handle = ctx.cudnn_workspace_handle();
  workspace_handle.RunFunc(
      [&](void* workspace_ptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnFlashAttnForward(
            cudnn,
            flashAttnDesc,
            flashAttnInfo,
            q_desc,
            k_desc,
            v_desc,
            o_desc,
            nullptr,
            lse_desc,
            q_padded.data(),
            k_padded.data(),
            v_padded.data(),
            nullptr,
            is_unpad ? cu_seqlens_q.data<int>() : nullptr,
            is_unpad ? cu_seqlens_k.data<int>() : nullptr,
            nullptr,  // reinterpret_cast<int *>(d_loWinIdx.data_ptr()),
            nullptr,  // reinterpret_cast<int *>(d_hiWinIdx.data_ptr()),
            nullptr,
            workspace_ptr,
            out->data(),
            softmax_lse->data<float>()));
      },
      size_tmpbuf);

  out->Resize({total_q, num_heads, head_size});

  phi::dynload::cudnnDestroyFlashAttnDescriptor(flashAttnDesc);
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(q_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(k_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(v_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(o_desc));
  flashAttnDesc = nullptr;
  q_desc = nullptr;
  k_desc = nullptr;
  v_desc = nullptr;
  o_desc = nullptr;
}

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
    const Scalar& max_seqlen_q,
    const Scalar& max_seqlen_k,
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
  FlashAttnUnpaddedBaseKernel<T>(ctx,
                                 q,
                                 k,
                                 v,
                                 cu_seqlens_q,
                                 cu_seqlens_k,
                                 fixed_seed_offset,
                                 attn_mask,
                                 max_seqlen_q,
                                 max_seqlen_k,
                                 scale,
                                 dropout,
                                 causal,
                                 return_softmax,
                                 is_test,
                                 rng_name,
                                 out,
                                 softmax,
                                 softmax_lse,
                                 seed_offset,
                                 false /*varlen_padded*/);
}

static void sliceFlattenView(const DenseTensor& in,
                             DenseTensor* out,
                             int axis,
                             int64_t offset,
                             int64_t sliceLength) {
  PADDLE_ENFORCE_LT(
      axis,
      in.dims().size(),
      phi::errors::InvalidArgument("sliceView receive axis out of bound"));
  std::array<int64_t, DDim::kMaxRank> dimArr;
  std::array<int64_t, DDim::kMaxRank> strideArr;
  auto id = dimArr.begin(), is = strideArr.begin();
  for (int i = 0; i < in.dims().size(); i++) {
    if (i == axis) continue;
    if (i == axis + 1)
      *id = in.dims()[i] * sliceLength;
    else
      *id = in.dims()[i];
    *is = in.strides()[i];
    id++;
    is++;
  }
  *out = DenseTensor{
      in.Holder(),
      DenseTensorMeta{in.dtype(),
                      DDim{dimArr.data(), in.dims().size() - 1},
                      DDim(strideArr.data(), in.dims().size() - 1)}};
  out->set_offset(in.offset() +
                  offset * in.strides()[axis] * SizeOf(out->dtype()));
}

template <typename T, typename Context>
void FlashAttnVarlenQKVPackedKernel(
    const Context& ctx,
    const DenseTensor& qkv,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float scale,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    bool varlen_padded,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
  RaiseNotSupportedError();
}

template <typename T, typename Context>
void FlashAttnBaseKernel(
    const Context& ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const paddle::optional<DenseTensor>& attn_mask_start_row_indices,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    int attn_mask_start_row,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(k.dims().size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(v.dims().size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(out->dims().size(),
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

  // ixdnn
  PADDLE_ENFORCE_LE(head_size,
                    512,
                    phi::errors::InvalidArgument(
                        "FlashAttention forward only supports head dimension "
                        "at most 512, but get head_dim=[head_size]"));
  PADDLE_ENFORCE_EQ(
      num_heads % num_heads_k,
      0,
      phi::errors::InvalidArgument(
          "Number of heads in key/value must divide number of heads in query"));

  int64_t total_q = batch_size * seqlen_q;
  if (is_test) dropout = 0.0f;
  if (!out->IsInitialized()) ctx.template Alloc<T>(out);

  softmax_lse->Resize({batch_size, num_heads, seqlen_q});
  ctx.template Alloc<float>(softmax_lse);

  if (return_softmax) {
    softmax->Resize({batch_size, num_heads, seqlen_q, seqlen_k});
    ctx.template Alloc<T>(softmax);
  }

  DenseTensor q_padded, k_padded, v_padded;
  const int64_t head_size_rounded = head_size + 32 - head_size % 32;
  q_padded = q;
  k_padded = k;
  v_padded = v;

  int64_t seed;
  int64_t offset;

  if (fixed_seed_offset.get_ptr()) {
    const int64_t* fixed_seed_offset_data =
        fixed_seed_offset.get_ptr()->data<int64_t>();
    seed = static_cast<int64_t>(fixed_seed_offset_data[0]);
    offset = static_cast<int64_t>(fixed_seed_offset_data[1]);
  } else {
    uint64_t inc = batch_size * num_heads * 64;
    std::pair<int64_t, int64_t> seed_offset_pair;
    if (rng_name != "") {
      auto gen = phi::GetRandomSeedGenerator(rng_name);
      seed_offset_pair = gen->IncrementOffset(inc);
    } else {
      auto* gen = ctx.GetGenerator();
      seed_offset_pair = gen->IncrementOffset(inc);
    }
    seed = seed_offset_pair.first;
    offset = seed_offset_pair.second;
  }
  PhiloxCudaState philox_state = PhiloxCudaState(seed, offset);

  seed_offset->Resize({2});
  int64_t* seed_offset_data = ctx.template HostAlloc<int64_t>(seed_offset);
  seed_offset_data[0] = static_cast<int64_t>(seed);
  seed_offset_data[1] = static_cast<int64_t>(offset);

  cudnnFlashAttnConfigInfo flashAttnInfo;
  flashAttnInfo.softmax_scale = std::sqrt(1.f / head_size);
  flashAttnInfo.dropout_prob = dropout;
  flashAttnInfo.is_causal = causal;
  // flashAttnInfo.is_alibi              = use_alibi;
  // flashAttnInfo.alibi_mode            = alibi_mode;
  flashAttnInfo.return_softmax_lse = true;
  flashAttnInfo.philox_args =
      *(reinterpret_cast<cudnnPhiloxCudaState*>(&philox_state));
  flashAttnInfo.imp_mode = FLAGS_ixdnn_imp_mode ? CUDNN_FATTN_LEAST_MEM_MODE
                                                : CUDNN_FATTN_BALANCE_MODE;
  flashAttnInfo.is_unpad = false;
  flashAttnInfo.batch = batch_size;
  flashAttnInfo.max_seq_len_src = seqlen_q;
  flashAttnInfo.max_seq_len_trg = seqlen_k;

  int32_t nb_dims = 4;
  std::vector<int32_t> qShape, kShape, vShape, oShape, lseShape;
  std::vector<int32_t> qStride, kStride, vStride, oStride, lseStride;

  cudnnDataType_t dataType;
  if (q.dtype() == phi::DataType::FLOAT16) {
    dataType = CUDNN_DATA_HALF;
  } else if (q.dtype() == phi::DataType::BFLOAT16) {
    dataType = CUDNN_DATA_BFLOAT16;
  } else if (q.dtype() == phi::DataType::FLOAT32) {
    dataType = CUDNN_DATA_FLOAT;
  }

  cudnnHandle_t cudnn;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnCreate(&cudnn));

  cudnnFlashAttnDescriptor_t flashAttnDesc;
  cudnnTensorDescriptor_t q_desc;
  cudnnTensorDescriptor_t k_desc;
  cudnnTensorDescriptor_t v_desc;
  cudnnTensorDescriptor_t o_desc;
  cudnnTensorDescriptor_t lse_desc;
  cudnnTensorDescriptor_t m_desc;

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateFlashAttnDescriptor(&flashAttnDesc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&q_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&k_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&v_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&o_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&lse_desc));

  copyDimsAndStrides(q_padded, qShape, qStride);
  copyDimsAndStrides(k_padded, kShape, kStride);
  copyDimsAndStrides(v_padded, vShape, vStride);
  copyDimsAndStrides(out, oShape, oStride);
  copyDimsAndStrides(softmax_lse, lseShape, lseStride);

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      q_desc, dataType, nb_dims, qShape.data(), qStride.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      k_desc, dataType, nb_dims, kShape.data(), kStride.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      v_desc, dataType, nb_dims, vShape.data(), vStride.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      o_desc, dataType, nb_dims, oShape.data(), oStride.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnSetTensorNdDescriptor(lse_desc,
                                               CUDNN_DATA_FLOAT,
                                               nb_dims - 1,
                                               lseShape.data(),
                                               lseStride.data()));

  if (attn_mask.get_ptr()) {
    PADDLE_ENFORCE_NE(causal,
                      true,
                      phi::errors::InvalidArgument(
                          "When attn_mask is set, causal can not be true."));

    PADDLE_ENFORCE_EQ(
        (attn_mask.get_ptr())->dtype(),
        q.dtype(),
        phi::errors::InvalidArgument(
            "attn_mask is expected to have the same data type with q."));

    std::vector<int32_t> mShape;
    std::vector<int32_t> mStride;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateTensorDescriptor(&m_desc));
    copyDimsAndStrides(attn_mask.get(), mShape, mStride);
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        m_desc, dataType, nb_dims, mShape.data(), mStride.data()));
  }

  size_t size_tmpbuf = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnGetFlashAttnBuffers(cudnn,
                                             flashAttnDesc,
                                             batch_size,
                                             num_heads,
                                             seqlen_q,
                                             seqlen_k,
                                             head_size,
                                             total_q,
                                             false,
                                             false,
                                             &size_tmpbuf,
                                             flashAttnInfo.imp_mode));

  auto workspace_handle = ctx.cudnn_workspace_handle();
  workspace_handle.RunFunc(
      [&](void* workspace_ptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnFlashAttnForward(
            cudnn,
            flashAttnDesc,
            flashAttnInfo,
            q_desc,
            k_desc,
            v_desc,
            o_desc,
            attn_mask.get_ptr() ? m_desc : nullptr,
            lse_desc,
            q_padded.data(),
            k_padded.data(),
            v_padded.data(),
            attn_mask.get_ptr() ? (attn_mask.get_ptr())->data() : nullptr,
            nullptr,  // reinterpret_cast<int *>(cu_seqlens_q.data_ptr()),
            nullptr,  // reinterpret_cast<int *>(cu_seqlens_k.data_ptr()),
            nullptr,  // reinterpret_cast<int *>(d_loWinIdx.data_ptr()),
            nullptr,  // reinterpret_cast<int *>(d_hiWinIdx.data_ptr()),
            nullptr,
            workspace_ptr,
            out->data(),
            softmax_lse->data<float>()));
      },
      size_tmpbuf);

  phi::dynload::cudnnDestroyFlashAttnDescriptor(flashAttnDesc);
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(q_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(k_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(v_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(o_desc));
  flashAttnDesc = nullptr;
  q_desc = nullptr;
  k_desc = nullptr;
  v_desc = nullptr;
  o_desc = nullptr;
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
  FlashAttnBaseKernel<T, Context>(ctx,
                                  q,
                                  k,
                                  v,
                                  fixed_seed_offset,
                                  attn_mask,
                                  paddle::none,
                                  dropout,
                                  causal,
                                  return_softmax,
                                  is_test,
                                  rng_name,
                                  0,
                                  out,
                                  softmax,
                                  softmax_lse,
                                  seed_offset);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(flash_attn_unpadded,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnUnpaddedKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_CUSTOM_KERNEL_REGISTER(flash_attn,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}
