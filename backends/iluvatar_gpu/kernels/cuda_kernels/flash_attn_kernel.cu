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
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/utils/none.h"

#ifdef PADDLE_WITH_FLASHATTN_V3
#include "paddle/phi/kernels/gpu/flash_attn_v3_kernel.h"
#endif

COMMON_DECLARE_int32(flash_attn_version);
COMMON_DECLARE_bool(cudnn_deterministic);

#ifdef PADDLE_WITH_COREX
COMMON_DECLARE_bool(enable_ixattnbkd);
COMMON_DECLARE_int32(imp_mode);
#endif

namespace phi {
template <typename OutT>
struct ZeroFunctor {
  __device__ __forceinline__ OutT operator()() const {
    return static_cast<OutT>(0);
  }
};

template <typename T, typename Context>
void FlashAttnUnpaddedBaseKernel(
    const Context& dev_ctx,
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
  if (!out->IsInitialized()) dev_ctx.template Alloc<T>(out);
  if (varlen_padded) {
    std::vector<const DenseTensor*> inputs{};
    std::vector<DenseTensor*> outputs{out};

    phi::funcs::ElementwiseKernel<T>(
        dev_ctx, inputs, &outputs, ZeroFunctor<T>());
  }

  // q, k, v [total_q/k/v, num_heads, head_dim]
  auto dims = q.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      3,
      common::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                      "[total_seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      k.dims().size(),
      3,
      common::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                      "[total_seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      v.dims().size(),
      3,
      common::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                      "[total_seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      out->dims().size(),
      3,
      common::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                      "[total_seq_len, num_heads, head_dim]"));

  PADDLE_ENFORCE_EQ(q.strides()[q.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(k.strides()[k.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(v.strides()[v.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size = dims[2];
  const int64_t num_heads_k = k.dims()[1];
  const int64_t total_q = dims[0];
  const int head_size_v = v.dims()[2];

  int64_t max_seqlen_q = max_seqlen_q_.to<int64_t>();
  int64_t max_seqlen_k = max_seqlen_k_.to<int64_t>();

  int64_t seed;
  int64_t offset;
  if (dropout > 0.0f) {
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
        auto* gen = dev_ctx.GetGenerator();
        seed_offset_pair = gen->IncrementOffset(inc);
      }
      seed = seed_offset_pair.first;
      offset = seed_offset_pair.second;
    }
  } else {
    seed = 0;
    offset = 0;
  }

  PhiloxCudaState philox_state = PhiloxCudaState(seed, offset);

  seed_offset->Resize({2});
  int64_t* seed_offset_data = dev_ctx.template HostAlloc<int64_t>(seed_offset);
  seed_offset_data[0] = static_cast<int64_t>(seed);
  seed_offset_data[1] = static_cast<int64_t>(offset);

  PADDLE_ENFORCE_GE(
      batch_size,
      0,
      phi::errors::InvalidArgument("batch size must be positive"));
  PADDLE_ENFORCE_EQ(
      num_heads % num_heads_k,
      0,
      phi::errors::InvalidArgument("Number of heads in key/value must divide "
                                   "number of heads in query"));

  if (!out->IsInitialized()) dev_ctx.template Alloc<T>(out);

  softmax_lse->Resize({num_heads, total_q});
  dev_ctx.template Alloc<float>(softmax_lse);

  bool accuracy_first = false;
  if (attn_mask) {
    causal = false;
    phi::DenseTensor min_tensor;
    min_tensor.Resize({1});
    dev_ctx.template Alloc<T>(&min_tensor);

    std::vector<int> reduce_dims;
    for (int64_t i = 0; i < attn_mask->dims().size(); ++i) {
      reduce_dims.push_back(i);
    }
    funcs::ReduceKernel<T, T, kps::MinFunctor, kps::IdentityFunctor<T>>(
        dev_ctx,
        *attn_mask,
        &min_tensor,
        kps::IdentityFunctor<T>(),
        reduce_dims);

    std::vector<T> host_min;
    TensorToVector(min_tensor, dev_ctx, &host_min);

    float min_val = static_cast<float>(host_min[0]);
    constexpr float threshold = -3.3895313892515355e+37f;
    accuracy_first = (min_val < threshold);
    VLOG(2) << "flash_attn attn_mask accuracy_first: " << accuracy_first
            << ", causal: " << causal;
  }

  if (FLAGS_enable_ixattnbkd) {
    // ixattnbkd unpad
    ixAttnBkdConfigInfo ixAttnbkdInfo;
    ixAttnbkdInfo.stream = dev_ctx.stream();
    ixAttnbkdInfo.softmax_scale = std::sqrt(1.f / head_size);
    ixAttnbkdInfo.dropout_prob = dropout;
    ixAttnbkdInfo.is_causal = causal;
    ixAttnbkdInfo.causal_mode = 0;
    // ixAttnbkdInfo.is_alibi = use_alibi;
    // ixAttnbkdInfo.alibi_mode = alibi_mode;
    ixAttnbkdInfo.return_softmax_lse = true;
    ixAttnbkdInfo.is_unpad = true;
    ixAttnbkdInfo.batch = batch_size;
    ixAttnbkdInfo.max_seq_len_src = max_seqlen_q;
    ixAttnbkdInfo.max_seq_len_trg = max_seqlen_k;
    ixAttnbkdInfo.imp_mode =
        FLAGS_imp_mode ? IXATTNBKD_FATTN_MEM_MODE : IXATTNBKD_FATTN_PERF_MODE;
    ixAttnbkdInfo.accuracy_first = accuracy_first;

    ixAttnBkdDataType_t dataType;
    if (q.dtype() == phi::DataType::FLOAT16) {
      dataType = IXATTNBKD_DATA_HALF;
    } else if (q.dtype() == phi::DataType::BFLOAT16) {
      dataType = IXATTNBKD_DATA_BF16;
    } else if (q.dtype() == phi::DataType::FLOAT32) {
      dataType = IXATTNBKD_DATA_FLOAT;
    }
    ixAttnBkdTensorDesc q_desc, k_desc, v_desc, o_desc, m_desc, lse_desc;
    SetIxAttnBkdTensor(&q_desc, q, dataType);
    SetIxAttnBkdTensor(&k_desc, k, dataType);
    SetIxAttnBkdTensor(&v_desc, v, dataType);
    SetIxAttnBkdTensor(&o_desc, out, dataType);
    if (attn_mask) {
      PADDLE_ENFORCE_NE(causal,
                        true,
                        phi::errors::InvalidArgument(
                            "When attn_mask is set, causal can not be true."));

      PADDLE_ENFORCE_EQ(
          (attn_mask.get_ptr())->dtype(),
          q.dtype(),
          phi::errors::InvalidArgument(
              "attn_mask is expected to have the same data type with q."));

      SetIxAttnBkdTensor(&m_desc, attn_mask.get_ptr(), dataType);
    }
    SetIxAttnBkdTensor(&lse_desc, softmax_lse, IXATTNBKD_DATA_FLOAT);

    size_t size_tmpbuf = 0;
    PADDLE_IXATTNBKD_CHECK(
        ixAttnBkdGetFlashAttnFwdBuffSize(batch_size,
                                         num_heads,
                                         num_heads_k,
                                         max_seqlen_q,
                                         max_seqlen_k,
                                         head_size,
                                         head_size_v,
                                         &size_tmpbuf,
                                         ixAttnbkdInfo.imp_mode));

    auto workspace =
        phi::Empty<uint8_t>(dev_ctx, {static_cast<int64_t>(size_tmpbuf)});
    PADDLE_IXATTNBKD_CHECK(ixAttnBkdFlashAttnForward(
        ixAttnbkdInfo,
        q_desc,
        k_desc,
        v_desc,
        m_desc,
        o_desc,
        lse_desc,
        q.data(),
        k.data(),
        v.data(),
        attn_mask.get_ptr() ? (attn_mask.get_ptr())->data() : nullptr,
        nullptr,
        nullptr,
        cu_seqlens_q.data<int>(),
        cu_seqlens_k.data<int>(),
        workspace.data(),
        out->data(),
        softmax_lse->data<float>()));
  } else {
    // ixdnn unpad
    cudnnFlashAttnConfigInfo flashAttnInfo;
    flashAttnInfo.softmax_scale = std::sqrt(1.f / head_size);
    flashAttnInfo.dropout_prob = dropout;
    flashAttnInfo.is_causal = causal;
    flashAttnInfo.causal_mode = 0;
    // flashAttnInfo.is_alibi = use_alibi;
    // flashAttnInfo.alibi_mode = alibi_mode;
    flashAttnInfo.return_softmax_lse = true;
    flashAttnInfo.philox_args =
        *(reinterpret_cast<cudnnPhiloxCudaState*>(&philox_state));
    flashAttnInfo.is_unpad = true;
    flashAttnInfo.batch = batch_size;
    flashAttnInfo.max_seq_len_src = max_seqlen_q;
    flashAttnInfo.max_seq_len_trg = max_seqlen_k;
    flashAttnInfo.imp_mode =
        FLAGS_imp_mode ? CUDNN_FATTN_LEAST_MEM_MODE : CUDNN_FATTN_BALANCE_MODE;
    flashAttnInfo.accuracy_first = accuracy_first;

    int32_t nb_dims = 3;
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

    cudnnHandle_t cudnn = dev_ctx.cudnn_handle();
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

    copyDimsAndStrides(q, qShape, qStride);
    copyDimsAndStrides(k, kShape, kStride);
    copyDimsAndStrides(v, vShape, vStride);
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
        phi::dynload::cudnnSetTensorNdDescriptor_lowerbound_2(
            lse_desc,
            CUDNN_DATA_FLOAT,
            nb_dims - 1,
            lseShape.data(),
            lseStride.data()));

    if (attn_mask) {
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
                                               max_seqlen_q,
                                               max_seqlen_k,
                                               head_size,
                                               total_q,
                                               true,
                                               false,
                                               &size_tmpbuf,
                                               flashAttnInfo.imp_mode));

    auto workspace =
        phi::Empty<uint8_t>(dev_ctx, {static_cast<int64_t>(size_tmpbuf)});
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
        q.data(),
        k.data(),
        v.data(),
        attn_mask.get_ptr() ? (attn_mask.get_ptr())->data() : nullptr,
        cu_seqlens_q.data<int>(),
        cu_seqlens_k.data<int>(),
        nullptr,
        nullptr,
        nullptr,
        workspace.data(),
        out->data(),
        softmax_lse->data<float>()));

    cudaDeviceSynchronize();

    phi::dynload::cudnnDestroyFlashAttnDescriptor(flashAttnDesc);
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnDestroyTensorDescriptor(q_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnDestroyTensorDescriptor(k_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnDestroyTensorDescriptor(v_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnDestroyTensorDescriptor(o_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnDestroyTensorDescriptor(lse_desc));
    if (attn_mask) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnDestroyTensorDescriptor(m_desc));
    }
    flashAttnDesc = nullptr;
    q_desc = nullptr;
    k_desc = nullptr;
    v_desc = nullptr;
    o_desc = nullptr;
    lse_desc = nullptr;
    if (attn_mask) {
      m_desc = nullptr;
    }
  }
}

template <typename T, typename Context>
void FlashAttnUnpaddedKernel(
    const Context& dev_ctx,
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
#if defined(PADDLE_WITH_FLASHATTN) || defined(PADDLE_WITH_COREX)
  FlashAttnUnpaddedBaseKernel<T>(dev_ctx,
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
#else
  RaiseNotSupportedError();
#endif
}

static void sliceFlattenView(const DenseTensor& in,
                             DenseTensor* out,
                             int axis,
                             int64_t offset,
                             int64_t sliceLength) {
  PADDLE_ENFORCE_LT(
      axis,
      in.dims().size(),
      common::errors::InvalidArgument("sliceView receive axis out of bound"));
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
    const Context& dev_ctx,
    const DenseTensor& qkv,
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
    bool varlen_padded,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
#if defined(PADDLE_WITH_FLASHATTN) || defined(PADDLE_WITH_COREX)
  const auto head_groupnum = qkv.dims()[1];  // nheads/nheads_k + 1 + 1
  DenseTensor q, k, v;
  sliceFlattenView(qkv, &q, 1, 0, head_groupnum - 2);
  sliceFlattenView(qkv, &k, 1, head_groupnum - 2, 1);
  sliceFlattenView(qkv, &v, 1, head_groupnum - 1, 1);
  FlashAttnUnpaddedBaseKernel<T>(dev_ctx,
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
                                 varlen_padded);
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnBaseKernel(
    const Context& dev_ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const paddle::optional<DenseTensor>& startend_row_indices,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    common::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(k.dims().size(),
                    4,
                    common::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(v.dims().size(),
                    4,
                    common::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(out->dims().size(),
                    4,
                    common::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  PADDLE_ENFORCE_EQ(q.strides()[q.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(k.strides()[k.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(v.strides()[v.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];

  const float softmax_scale = 1.0f / std::sqrt(head_size);
  const float softmax_unscale = std::sqrt(head_size);
  dropout = is_test ? 0.0f : dropout;
  const int64_t total_q = batch_size * seqlen_q;
  const int64_t head_size_v = v.dims()[3];

  int64_t seed;
  int64_t offset;
  if (dropout > 0.0f) {
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
        auto* gen = dev_ctx.GetGenerator();
        seed_offset_pair = gen->IncrementOffset(inc);
      }
      seed = seed_offset_pair.first;
      offset = seed_offset_pair.second;
    }
  } else {
    seed = 0;
    offset = 0;
  }

  PhiloxCudaState philox_state = PhiloxCudaState(seed, offset);
  seed_offset->Resize({2});
  int64_t* seed_offset_data = dev_ctx.template HostAlloc<int64_t>(seed_offset);
  seed_offset_data[0] = static_cast<int64_t>(seed);
  seed_offset_data[1] = static_cast<int64_t>(offset);

  PADDLE_ENFORCE_GE(
      batch_size,
      0,
      phi::errors::InvalidArgument("batch size must be positive"));
  PADDLE_ENFORCE_EQ(
      num_heads % num_heads_k,
      0,
      phi::errors::InvalidArgument("Number of heads in key/value must divide "
                                   "number of heads in query"));

  if (!out->IsInitialized()) dev_ctx.template Alloc<T>(out);

  softmax_lse->Resize({batch_size, num_heads, seqlen_q});
  dev_ctx.template Alloc<float>(softmax_lse);

  bool accuracy_first = false;
  if (attn_mask) {
    causal = false;
    phi::DenseTensor min_tensor;
    min_tensor.Resize({1});
    dev_ctx.template Alloc<T>(&min_tensor);

    std::vector<int> reduce_dims;
    for (int64_t i = 0; i < attn_mask->dims().size(); ++i) {
      reduce_dims.push_back(i);
    }
    funcs::ReduceKernel<T, T, kps::MinFunctor, kps::IdentityFunctor<T>>(
        dev_ctx,
        *attn_mask,
        &min_tensor,
        kps::IdentityFunctor<T>(),
        reduce_dims);

    std::vector<T> host_min;
    TensorToVector(min_tensor, dev_ctx, &host_min);

    float min_val = static_cast<float>(host_min[0]);
    constexpr float threshold = -3.3895313892515355e+37f;
    accuracy_first = (min_val < threshold);
    VLOG(2) << "flash_attn attn_mask accuracy_first: " << accuracy_first
            << ", causal: " << causal;
  }

  if (FLAGS_enable_ixattnbkd) {
    // ixattnbkd
    ixAttnBkdConfigInfo ixAttnbkdInfo;
    ixAttnbkdInfo.stream = dev_ctx.stream();
    ixAttnbkdInfo.softmax_scale = std::sqrt(1.f / head_size);
    ixAttnbkdInfo.dropout_prob = dropout;
    ixAttnbkdInfo.is_causal = causal;
    ixAttnbkdInfo.causal_mode = 0;
    // ixAttnbkdInfo.is_alibi = use_alibi;
    // ixAttnbkdInfo.alibi_mode = alibi_mode;
    ixAttnbkdInfo.return_softmax_lse = true;
    ixAttnbkdInfo.is_unpad = false;
    ixAttnbkdInfo.batch = batch_size;
    ixAttnbkdInfo.max_seq_len_src = seqlen_q;
    ixAttnbkdInfo.max_seq_len_trg = seqlen_k;
    ixAttnbkdInfo.imp_mode =
        FLAGS_imp_mode ? IXATTNBKD_FATTN_MEM_MODE : IXATTNBKD_FATTN_PERF_MODE;
    ixAttnbkdInfo.accuracy_first = accuracy_first;

    ixAttnBkdDataType_t dataType;
    if (q.dtype() == phi::DataType::FLOAT16) {
      dataType = IXATTNBKD_DATA_HALF;
    } else if (q.dtype() == phi::DataType::BFLOAT16) {
      dataType = IXATTNBKD_DATA_BF16;
    } else if (q.dtype() == phi::DataType::FLOAT32) {
      dataType = IXATTNBKD_DATA_FLOAT;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "flash_attn_bwd receive input with dtype == %s",
          phi::DataTypeToString(q.dtype())));
    }
    ixAttnBkdTensorDesc q_desc, k_desc, v_desc, o_desc, m_desc, lse_desc;
    SetIxAttnBkdTensor(&q_desc, q, dataType);
    SetIxAttnBkdTensor(&k_desc, k, dataType);
    SetIxAttnBkdTensor(&v_desc, v, dataType);
    SetIxAttnBkdTensor(&o_desc, out, dataType);
    if (attn_mask) {
      PADDLE_ENFORCE_NE(causal,
                        true,
                        phi::errors::InvalidArgument(
                            "When attn_mask is set, causal can not be true."));

      PADDLE_ENFORCE_EQ(
          (attn_mask.get_ptr())->dtype(),
          q.dtype(),
          phi::errors::InvalidArgument(
              "attn_mask is expected to have the same data type with q."));

      SetIxAttnBkdTensor(&m_desc, attn_mask.get_ptr(), dataType);
    }
    SetIxAttnBkdTensor(&lse_desc, softmax_lse, IXATTNBKD_DATA_FLOAT);

    size_t size_tmpbuf = 0;
    PADDLE_IXATTNBKD_CHECK(
        ixAttnBkdGetFlashAttnFwdBuffSize(batch_size,
                                         num_heads,
                                         num_heads_k,
                                         seqlen_q,
                                         seqlen_k,
                                         head_size,
                                         head_size_v,
                                         &size_tmpbuf,
                                         ixAttnbkdInfo.imp_mode));

    auto workspace =
        phi::Empty<uint8_t>(dev_ctx, {static_cast<int64_t>(size_tmpbuf)});
    PADDLE_IXATTNBKD_CHECK(ixAttnBkdFlashAttnForward(
        ixAttnbkdInfo,
        q_desc,
        k_desc,
        v_desc,
        m_desc,
        o_desc,
        lse_desc,
        q.data(),
        k.data(),
        v.data(),
        attn_mask.get_ptr() ? (attn_mask.get_ptr())->data() : nullptr,
        nullptr,  // reinterpret_cast<int *>(cu_seqlens_q.data_ptr()),
        nullptr,  // reinterpret_cast<int *>(cu_seqlens_k.data_ptr()),
        nullptr,  // reinterpret_cast<int *>(d_loWinIdx.data_ptr()),
        nullptr,  // reinterpret_cast<int *>(d_hiWinIdx.data_ptr()),
        workspace.data(),
        out->data(),
        softmax_lse->data<float>()));

  } else {
    // ixdnn
    cudnnFlashAttnConfigInfo flashAttnInfo;
    flashAttnInfo.softmax_scale = std::sqrt(1.f / head_size);
    flashAttnInfo.dropout_prob = dropout;
    flashAttnInfo.is_causal = causal;
    flashAttnInfo.causal_mode = 0;
    // flashAttnInfo.is_alibi = use_alibi;
    // flashAttnInfo.alibi_mode = alibi_mode;
    flashAttnInfo.return_softmax_lse = true;
    flashAttnInfo.philox_args =
        *(reinterpret_cast<cudnnPhiloxCudaState*>(&philox_state));
    flashAttnInfo.is_unpad = false;
    flashAttnInfo.batch = batch_size;
    flashAttnInfo.max_seq_len_src = seqlen_q;
    flashAttnInfo.max_seq_len_trg = seqlen_k;
    flashAttnInfo.imp_mode =
        FLAGS_imp_mode ? CUDNN_FATTN_LEAST_MEM_MODE : CUDNN_FATTN_BALANCE_MODE;
    flashAttnInfo.accuracy_first = accuracy_first;

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

    cudnnHandle_t cudnn = dev_ctx.cudnn_handle();
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

    copyDimsAndStrides(q, qShape, qStride);
    copyDimsAndStrides(k, kShape, kStride);
    copyDimsAndStrides(v, vShape, vStride);
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

    if (attn_mask) {
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

    auto workspace =
        phi::Empty<uint8_t>(dev_ctx, {static_cast<int64_t>(size_tmpbuf)});

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
        q.data(),
        k.data(),
        v.data(),
        attn_mask.get_ptr() ? (attn_mask.get_ptr())->data() : nullptr,
        nullptr,  // reinterpret_cast<int *>(cu_seqlens_q.data_ptr()),
        nullptr,  // reinterpret_cast<int *>(cu_seqlens_k.data_ptr()),
        nullptr,  // reinterpret_cast<int *>(d_loWinIdx.data_ptr()),
        nullptr,  // reinterpret_cast<int *>(d_hiWinIdx.data_ptr()),
        nullptr,
        workspace.data(),
        out->data(),
        softmax_lse->data<float>()));

    cudaDeviceSynchronize();

    phi::dynload::cudnnDestroyFlashAttnDescriptor(flashAttnDesc);
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnDestroyTensorDescriptor(q_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnDestroyTensorDescriptor(k_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnDestroyTensorDescriptor(v_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnDestroyTensorDescriptor(o_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnDestroyTensorDescriptor(lse_desc));
    if (attn_mask) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnDestroyTensorDescriptor(m_desc));
    }
    flashAttnDesc = nullptr;
    q_desc = nullptr;
    k_desc = nullptr;
    v_desc = nullptr;
    o_desc = nullptr;
    lse_desc = nullptr;
    if (attn_mask) {
      m_desc = nullptr;
    }
  }
}

template <typename T, typename Context>
void FlashAttnKernel(const Context& dev_ctx,
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
  FlashAttnBaseKernel<T, Context>(dev_ctx,
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
                                  out,
                                  softmax,
                                  softmax_lse,
                                  seed_offset);
}

template <typename T, typename Context>
void FlashAttnQKVPackedKernel(
    const Context& dev_ctx,
    const DenseTensor& qkv,
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
#if defined(PADDLE_WITH_FLASHATTN) || defined(PADDLE_WITH_COREX)
  const auto head_groupnum = qkv.dims()[2];  // nheads/nheads_k + 1 + 1
  DenseTensor q, k, v;
  sliceFlattenView(qkv, &q, 2, 0, head_groupnum - 2);
  sliceFlattenView(qkv, &k, 2, head_groupnum - 2, 1);
  sliceFlattenView(qkv, &v, 2, head_groupnum - 1, 1);
  FlashAttnBaseKernel<T, Context>(dev_ctx,
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
                                  out,
                                  softmax,
                                  softmax_lse,
                                  seed_offset);
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashMaskKernel(const Context& dev_ctx,
                     const DenseTensor& q,
                     const DenseTensor& k,
                     const DenseTensor& v,
                     const DenseTensor& startend_row_indices,
                     const paddle::optional<DenseTensor>& fixed_seed_offset,
                     float dropout,
                     bool causal,
                     bool return_softmax,
                     bool is_test,
                     const std::string& rng_name,
                     DenseTensor* out,
                     DenseTensor* softmax,
                     DenseTensor* softmax_lse,
                     DenseTensor* seed_offset) {
#ifdef PADDLE_WITH_COREX
  RaiseNotSupportedError();
#else
  FlashAttnBaseKernel<T, Context>(dev_ctx,
                                  q,
                                  k,
                                  v,
                                  fixed_seed_offset,
                                  paddle::none,
                                  startend_row_indices,
                                  dropout,
                                  causal,
                                  return_softmax,
                                  is_test,
                                  rng_name,
                                  out,
                                  softmax,
                                  softmax_lse,
                                  seed_offset);
#endif
}

}  // namespace phi

PD_REGISTER_PLUGIN_KERNEL(flash_attn_unpadded,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnUnpaddedKernel,
#ifdef PADDLE_WITH_COREX
                          float,
#endif
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flash_attn_varlen_qkvpacked,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnVarlenQKVPackedKernel,
#ifdef PADDLE_WITH_COREX
                          float,
#endif
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flash_attn,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnKernel,
#ifdef PADDLE_WITH_COREX
                          float,
#endif
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flash_attn_qkvpacked,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnQKVPackedKernel,
#ifdef PADDLE_WITH_COREX
                          float,
#endif
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(1).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flashmask_attention,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashMaskKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(4).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}
