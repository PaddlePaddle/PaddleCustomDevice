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
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/flash_attn_grad_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"
#ifdef PADDLE_WITH_FLASHATTN_V3
#include "paddle/phi/kernels/gpu/flash_attn_v3_grad_kernel.h"
#endif

COMMON_DECLARE_bool(cudnn_deterministic);
COMMON_DECLARE_int32(flash_attn_version);

COMMON_DECLARE_bool(enable_ixattnbkd);
COMMON_DECLARE_int32(imp_mode);
namespace phi {

int get_num_split() {
  // 0 for an internal heuristic, which is optimal
  return FLAGS_cudnn_deterministic ? 1 : 0;
}

template <typename T, uint64_t HeaddimDiv32>
static __global__ void SumStridedKV(const T* src,
                                    T* dst,
                                    const uint64_t sRowDim1,
                                    const uint64_t sRowDim2,
                                    const uint64_t sRowDim3,
                                    const uint64_t sColDim,
                                    const uint64_t sRowStride1,
                                    const uint64_t sRowStride2,
                                    const uint64_t sColStride,
                                    const uint64_t dRowStride1,
                                    const uint64_t dRowStride2) {
  // SrcShape [seqlen, num_heads_k, num_heads/num_heads_k, headdim]
  // AxisName [row1  , row2       , col                  , row3   ]
  // LoopMap  [blockx, thready    , serialreduce         , threadx]
  // Ensure blockDim.x == 32 && blockDim.z == 1
  // Ensure sRowStride3 == dRowStride3 == 1 (headdim dim is contiguous)
  using IndexType = uint64_t;
  constexpr IndexType BlockDimX = 32;
  const IndexType SRow1Begin = blockIdx.x * sRowStride1;
  const IndexType SRow1End = sRowDim1 * sRowStride1;
  const IndexType SRow1Stride = gridDim.x * sRowStride1;

  const IndexType SRow2Begin = threadIdx.y * sRowStride2;
  const IndexType SRow2End = sRowDim2 * sRowStride2;
  const IndexType SRow2Stride = blockDim.y * sRowStride2;

  // const IndexType SRow3Begin = threadIdx.x * sRowStride3;
  // const IndexType SRow3End = sRowDim3 * sRowStride3;
  // const IndexType SRow3Stride = BlockDimX * sRowStride3;

  constexpr IndexType SColBegin = 0;
  const IndexType SColEnd = sColDim * sColStride;
  const IndexType SColStride = sColStride;

  const IndexType DRow1Begin = blockIdx.x * dRowStride1;
  const IndexType DRow1Stride = gridDim.x * dRowStride1;

  const IndexType DRow2Begin = threadIdx.y * dRowStride2;
  const IndexType DRow2Stride = dRowStride2;

  // const IndexType DRow3Begin = threadIdx.x * dRowStride3;
  // const IndexType DRow3Stride = blockDim.x * dRowStride3;

  for (auto row1 = SRow1Begin, drow1 = DRow1Begin; row1 < SRow1End;
       row1 += SRow1Stride, drow1 += DRow1Stride) {
    for (auto row2 = SRow2Begin, drow2 = DRow2Begin; row2 < SRow2End;
         row2 += SRow2Stride, drow2 += DRow2Stride) {
      const auto i1 = row1 + row2 + threadIdx.x;
      const auto di1 = drow1 + drow2 + threadIdx.x;
      T v[HeaddimDiv32];
#pragma unroll
      for (auto i = IndexType(0); i < HeaddimDiv32; i++) {
        v[i] = T{0};
      }
      for (auto col = SColBegin; col < SColEnd; col += SColStride) {
        const auto i2 = i1 + col;
#pragma unroll
        for (auto i = IndexType(0); i < HeaddimDiv32; i++) {
          v[i] += src[i2 + i * BlockDimX];
        }
      }
#pragma unroll
      for (auto i = IndexType(0); i < HeaddimDiv32; i++) {
        dst[di1 + i * BlockDimX] = v[i];
      }
    }
  }
}

template <typename T>
static auto selectSumkernel(int64_t headdim) {
  PADDLE_ENFORCE_LE(headdim,
                    256,
                    common::errors::InvalidArgument(
                        "FlashAttention only support headdim <= 256"));
  PADDLE_ENFORCE_EQ(headdim % 32,
                    0,
                    common::errors::InvalidArgument(
                        "FlashAttention only support headdim %% 32 == 0"));
  PADDLE_ENFORCE_NE(
      headdim, 0, common::errors::InvalidArgument("Headdim can't be zero"));
#define CASEN(n) \
  case n:        \
    return SumStridedKV<T, n>;
  switch (headdim / 32) {
    CASEN(1);
    CASEN(2);
    CASEN(3);
    CASEN(4);
    CASEN(5);
    CASEN(6);
    CASEN(7);
    CASEN(8);
  }
  PADDLE_FATAL("Unreachable in selectSumKernel");
#undef CASEN
}

template <typename T, typename Context>
static void kvReduceForGQA(const Context& dev_ctx,
                           const DenseTensor& dk_tmp,
                           DenseTensor* dk) {
  PADDLE_ENFORCE_EQ(
      dk->strides()[2],
      1,
      common::errors::InvalidArgument("headdim dimension must be contiguous"));
  PADDLE_ENFORCE_EQ(
      dk_tmp.strides()[3],
      1,
      common::errors::InvalidArgument("headdim dimension must be contiguous"));
  const int64_t reduceDimSize = dk_tmp.dims()[2];
  const size_t blockNum =
      std::min((static_cast<int64_t>(dk_tmp.dims()[0] + 31) / 32),
               static_cast<int64_t>(1024l));
  const dim3 threadNum{32, 4, 1};
  auto sumkernel = selectSumkernel<T>(dk_tmp.dims()[3]);
  sumkernel<<<blockNum, threadNum, 0, dev_ctx.stream()>>>(
      reinterpret_cast<const T*>(dk_tmp.data()),
      reinterpret_cast<T*>(dk->data()),
      dk_tmp.dims()[0],
      dk_tmp.dims()[1],
      dk_tmp.dims()[3],
      dk_tmp.dims()[2],
      dk_tmp.strides()[0],
      dk_tmp.strides()[1],
      // dk_tmp.strides()[3],
      dk_tmp.strides()[2],
      dk->strides()[0],
      dk->strides()[1]
      // dk->strides()[2]
  );
}
template <typename T, typename Context>
static void kvReduceBatchedForGQA(const Context& dev_ctx,
                                  const DenseTensor& dk_tmp,
                                  DenseTensor* dk) {
  PADDLE_ENFORCE_EQ(
      dk->strides()[3],
      1,
      common::errors::InvalidArgument("headdim dimension must be contiguous"));
  PADDLE_ENFORCE_EQ(
      dk_tmp.strides()[4],
      1,
      common::errors::InvalidArgument("headdim dimension must be contiguous"));
  PADDLE_ENFORCE_EQ(dk->strides()[0],
                    dk->strides()[1] * dk->dims()[1],
                    common::errors::InvalidArgument(
                        "batchsize dimension must be contiguous"));
  PADDLE_ENFORCE_EQ(dk_tmp.strides()[0],
                    dk_tmp.strides()[1] * dk_tmp.dims()[1],
                    common::errors::InvalidArgument(
                        "batchsize dimension must be contiguous"));
  const int64_t reduceDimSize = dk_tmp.dims()[3];
  const size_t blockNum = std::min(
      (static_cast<int64_t>(dk_tmp.dims()[0] * dk_tmp.dims()[1] + 31) / 32),
      static_cast<int64_t>(1024l));
  const dim3 threadNum{32, 4, 1};
  auto sumkernel = selectSumkernel<T>(dk_tmp.dims()[4]);
  // here implicitly flat [batch,seqlen], and require batch dim to be contiguous
  sumkernel<<<blockNum, threadNum, 0, dev_ctx.stream()>>>(
      reinterpret_cast<const T*>(dk_tmp.data()),
      reinterpret_cast<T*>(dk->data()),
      dk_tmp.dims()[0] * dk_tmp.dims()[1],
      dk_tmp.dims()[2],
      dk_tmp.dims()[4],
      dk_tmp.dims()[3],
      dk_tmp.strides()[1],
      dk_tmp.strides()[2],
      // dk_tmp.strides()[4],
      dk_tmp.strides()[3],
      dk->strides()[1],
      dk->strides()[2]
      // dk->strides()[3]
  );
}

template <typename T, typename Context>
void FlashAttnUnpaddedGradBaseKernel(
    const Context& dev_ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const DenseTensor& dout,
    const Scalar& max_seqlen_q_,
    const Scalar& max_seqlen_k_,
    float scale,
    float dropout,
    bool causal,
    DenseTensor* dq,
    DenseTensor* dk,
    DenseTensor* dv,
    bool varlen_padded) {
  auto dims = q.dims();

  const int64_t total_q = dims[0];
  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size_og = dout.dims()[2];
  const int64_t head_size = dims[2];
  const int64_t head_size_v = v.dims()[2];
  const int64_t total_k = k.dims()[0];
  const int64_t num_heads_k = k.dims()[1];

  bool is_mha = (num_heads == num_heads_k);

  DenseTensor* kdq = dq;
  DenseTensor dq_tmp;
  if (!dq) {
    dq_tmp.Resize(dims);
    dev_ctx.template Alloc<T>(&dq_tmp);
    kdq = &dq_tmp;
  }

  std::initializer_list<int64_t> dk_dv_shape = {
      total_k, num_heads_k, num_heads / num_heads_k, head_size};

  DenseTensor *kdk = dk, *kdv = dv;
  DenseTensor dk_tmp;
  if (!dk || !is_mha) {
    dk_tmp.Resize({total_k, num_heads, head_size});
    dev_ctx.template Alloc<T>(&dk_tmp);
    kdk = &dk_tmp;
  }

  DenseTensor dv_tmp;
  if (!dv || !is_mha) {
    dv_tmp.Resize({total_k, num_heads, head_size});
    dev_ctx.template Alloc<T>(&dv_tmp);
    kdv = &dv_tmp;
  }

  int num_splits = get_num_split();

  // TODO(umiswing): add shape check
  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      common::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  int64_t max_seqlen_q = max_seqlen_q_.to<int64_t>();
  int64_t max_seqlen_k = max_seqlen_k_.to<int64_t>();

  const int64_t* seed_offset_data = seed_offset.data<int64_t>();
  int64_t seed = static_cast<int64_t>(seed_offset_data[0]);
  int64_t offset = static_cast<int64_t>(seed_offset_data[1]);
  PhiloxCudaState philox_state = PhiloxCudaState(seed, offset);

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
    // ixattnbkd unpad bwd
    ixAttnBkdConfigInfo ixAttnbkdInfo;
    ixAttnbkdInfo.stream = dev_ctx.stream();
    ixAttnbkdInfo.softmax_scale = std::sqrt(1.f / head_size);
    ixAttnbkdInfo.dropout_prob = dropout;
    ixAttnbkdInfo.is_causal = causal;
    ixAttnbkdInfo.causal_mode = 0;
    // ixAttnbkdInfo.is_alibi              = use_alibi;
    // ixAttnbkdInfo.alibi_mode            = alibi_mode;
    ixAttnbkdInfo.return_softmax_lse = false;
    ixAttnbkdInfo.philox_args =
        *(reinterpret_cast<ixAttnBkdPhiloxState*>(&philox_state));
    ixAttnbkdInfo.imp_mode =
        FLAGS_imp_mode ? IXATTNBKD_FATTN_MEM_MODE : IXATTNBKD_FATTN_PERF_MODE;
    ixAttnbkdInfo.is_unpad = true;
    ixAttnbkdInfo.batch = batch_size;
    ixAttnbkdInfo.max_seq_len_src = max_seqlen_q;
    ixAttnbkdInfo.max_seq_len_trg = max_seqlen_k;
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

    ixAttnBkdTensorDesc q_desc, k_desc, v_desc, o_desc, m_desc, lse_desc,
        dq_desc, dk_desc, dv_desc, do_desc;
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
    SetIxAttnBkdTensor(&dq_desc, kdq, dataType);
    SetIxAttnBkdTensor(&dk_desc, kdk, dataType);
    SetIxAttnBkdTensor(&dv_desc, kdv, dataType);
    SetIxAttnBkdTensor(&do_desc, dout, dataType);

    size_t size_tmpbuf = 0;
    PADDLE_IXATTNBKD_CHECK(
        ixAttnBkdGetFlashAttnBwdBuffSize(batch_size,
                                         num_heads,
                                         num_heads_k,
                                         max_seqlen_q,
                                         max_seqlen_k,
                                         head_size,
                                         head_size_v,
                                         total_q,
                                         true,
                                         &size_tmpbuf,
                                         ixAttnbkdInfo.imp_mode));

    auto workspace =
        phi::Empty<uint8_t>(dev_ctx, {static_cast<int64_t>(size_tmpbuf)});
    PADDLE_IXATTNBKD_CHECK(ixAttnBkdFlashAttnBackward(
        ixAttnbkdInfo,
        q_desc,
        k_desc,
        v_desc,
        o_desc,
        m_desc,
        lse_desc,
        dq_desc,
        dk_desc,
        dv_desc,
        do_desc,
        q.data(),
        k.data(),
        v.data(),
        out.data(),
        dout.data(),
        attn_mask.get_ptr() ? (attn_mask.get_ptr())->data() : nullptr,
        nullptr,
        nullptr,
        cu_seqlens_q.data<int>(),
        cu_seqlens_k.data<int>(),
        softmax_lse.data<float>(),
        workspace.data(),
        kdq->data(),
        kdk->data(),
        kdv->data()));

  } else {
    // ixdnn unpad bwd
    cudnnFlashAttnConfigInfo flashAttnInfo;
    flashAttnInfo.softmax_scale = std::sqrt(1.f / head_size);
    flashAttnInfo.dropout_prob = dropout;
    flashAttnInfo.is_causal = causal;
    flashAttnInfo.causal_mode = 0;
    // flashAttnInfo.is_alibi              = use_alibi;
    // flashAttnInfo.alibi_mode            = alibi_mode;
    flashAttnInfo.return_softmax_lse = false;
    flashAttnInfo.philox_args =
        *(reinterpret_cast<cudnnPhiloxCudaState*>(&philox_state));
    flashAttnInfo.imp_mode =
        FLAGS_imp_mode ? CUDNN_FATTN_LEAST_MEM_MODE : CUDNN_FATTN_BALANCE_MODE;
    flashAttnInfo.is_unpad = true;
    flashAttnInfo.batch = batch_size;
    flashAttnInfo.max_seq_len_src = max_seqlen_q;
    flashAttnInfo.max_seq_len_trg = max_seqlen_k;
    flashAttnInfo.accuracy_first = accuracy_first;

    int32_t nb_dims = 3;
    std::vector<int32_t> qShape, kShape, vShape, oShape, lseShape, dqShape,
        dkShape, dvShape, doShape;
    std::vector<int32_t> qStride, kStride, vStride, oStride, lseStride,
        dqStride, dkStride, dvStride, doStride;

    cudnnDataType_t dataType;
    if (q.dtype() == phi::DataType::FLOAT16) {
      dataType = CUDNN_DATA_HALF;
    } else if (q.dtype() == phi::DataType::BFLOAT16) {
      dataType = CUDNN_DATA_BFLOAT16;
    } else if (q.dtype() == phi::DataType::FLOAT32) {
      dataType = CUDNN_DATA_FLOAT;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "flash_attn_bwd receive input with dtype == %s",
          phi::DataTypeToString(q.dtype())));
    }

    cudnnHandle_t cudnn;
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnCreate(&cudnn));

    cudnnFlashAttnDescriptor_t flashAttnDesc;
    cudnnTensorDescriptor_t q_desc, k_desc, v_desc, o_desc, m_desc, lse_desc,
        dq_desc, dk_desc, dv_desc, do_desc;

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
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateTensorDescriptor(&dq_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateTensorDescriptor(&dk_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateTensorDescriptor(&dv_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateTensorDescriptor(&do_desc));

    copyDimsAndStrides(q, qShape, qStride);
    copyDimsAndStrides(kdq, dqShape, dqStride);

    copyDimsAndStrides(k, kShape, kStride);
    copyDimsAndStrides(kdk, dkShape, dkStride);

    copyDimsAndStrides(v, vShape, vStride);
    copyDimsAndStrides(kdv, dvShape, dvStride);

    copyDimsAndStrides(out, oShape, oStride);
    copyDimsAndStrides(dout, doShape, doStride);
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
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        dq_desc, dataType, nb_dims, dqShape.data(), dqStride.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        dk_desc, dataType, nb_dims, dkShape.data(), dkStride.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        dv_desc, dataType, nb_dims, dvShape.data(), dvStride.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        do_desc, dataType, nb_dims, doShape.data(), doStride.data()));

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
                                               true,
                                               &size_tmpbuf,
                                               flashAttnInfo.imp_mode));

    auto workspace =
        phi::Empty<uint8_t>(dev_ctx, {static_cast<int64_t>(size_tmpbuf)});

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnFlashAttnBackward(
        cudnn,
        flashAttnDesc,
        flashAttnInfo,
        q_desc,
        k_desc,
        v_desc,
        o_desc,
        attn_mask.get_ptr() ? m_desc : nullptr,
        lse_desc,
        dq_desc,
        dk_desc,
        dv_desc,
        do_desc,
        q.data(),
        k.data(),
        v.data(),
        out.data(),
        dout.data(),
        softmax_lse.data<float>(),
        attn_mask.get_ptr() ? (attn_mask.get_ptr())->data() : nullptr,
        cu_seqlens_q.data<int>(),
        cu_seqlens_k.data<int>(),
        nullptr,
        nullptr,
        nullptr,
        workspace.data(),
        kdq->data(),
        kdk->data(),
        kdv->data()));

    auto destroy_tensor_desc = [&](cudnnTensorDescriptor_t desc) {
      if (desc) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnDestroyTensorDescriptor(desc));
      }
    };

    phi::dynload::cudnnDestroyFlashAttnDescriptor(flashAttnDesc);
    destroy_tensor_desc(q_desc);
    destroy_tensor_desc(k_desc);
    destroy_tensor_desc(v_desc);
    destroy_tensor_desc(o_desc);
    destroy_tensor_desc(lse_desc);
    if (attn_mask) {
      destroy_tensor_desc(m_desc);
    }
    destroy_tensor_desc(dq_desc);
    destroy_tensor_desc(dk_desc);
    destroy_tensor_desc(dv_desc);
    destroy_tensor_desc(do_desc);

    flashAttnDesc = nullptr;
    q_desc = nullptr;
    k_desc = nullptr;
    v_desc = nullptr;
    o_desc = nullptr;
    lse_desc = nullptr;
    if (attn_mask) {
      m_desc = nullptr;
    }
    dq_desc = nullptr;
    dk_desc = nullptr;
    dv_desc = nullptr;
    do_desc = nullptr;
  }
  if (!is_mha) {
    if (dk) {
      dk_tmp.Resize(dk_dv_shape);
      if (dk->meta().is_contiguous())
        phi::SumKernel<T, Context>(dev_ctx, dk_tmp, {2}, dk->type(), false, dk);
      else
        kvReduceForGQA<T, Context>(dev_ctx, dk_tmp, dk);
    }
    if (dv) {
      dv_tmp.Resize(dk_dv_shape);
      if (dv->meta().is_contiguous())
        phi::SumKernel<T, Context>(dev_ctx, dv_tmp, {2}, dv->type(), false, dv);
      else
        kvReduceForGQA<T, Context>(dev_ctx, dv_tmp, dv);
    }
  }
}

template <typename T, typename Context>
void FlashAttnUnpaddedGradKernel(const Context& dev_ctx,
                                 const DenseTensor& q,
                                 const DenseTensor& k,
                                 const DenseTensor& v,
                                 const DenseTensor& cu_seqlens_q,
                                 const DenseTensor& cu_seqlens_k,
                                 const DenseTensor& out,
                                 const DenseTensor& softmax_lse,
                                 const DenseTensor& seed_offset,
                                 const paddle::optional<DenseTensor>& attn_mask,
                                 const DenseTensor& dout,
                                 const Scalar& max_seqlen_q,
                                 const Scalar& max_seqlen_k,
                                 float scale,
                                 float dropout,
                                 bool causal,
                                 DenseTensor* dq,
                                 DenseTensor* dk,
                                 DenseTensor* dv) {
#if defined(PADDLE_WITH_FLASHATTN) || defined(PADDLE_WITH_COREX)
  if (dq) {
    dev_ctx.template Alloc<T>(dq);
  }
  if (dk) {
    dev_ctx.template Alloc<T>(dk);
  }
  if (dv) {
    dev_ctx.template Alloc<T>(dv);
  }
  FlashAttnUnpaddedGradBaseKernel<T>(dev_ctx,
                                     q,
                                     k,
                                     v,
                                     cu_seqlens_q,
                                     cu_seqlens_k,
                                     out,
                                     softmax_lse,
                                     seed_offset,
                                     attn_mask,
                                     dout,
                                     max_seqlen_q,
                                     max_seqlen_k,
                                     scale,
                                     dropout,
                                     causal,
                                     dq,
                                     dk,
                                     dv,
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
template <typename OutT>
struct ZeroFunctor {
  __device__ __forceinline__ OutT operator()() const {
    return static_cast<OutT>(0);
  }
};
template <typename T, typename Context>
void FlashAttnVarlenQKVPackedGradKernel(
    const Context& dev_ctx,
    const DenseTensor& qkv,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const DenseTensor& dout,
    const Scalar& max_seqlen_q,
    const Scalar& max_seqlen_k,
    float scale,
    float dropout,
    bool causal,
    bool varlen_padded,
    DenseTensor* dqkv) {
#if defined(PADDLE_WITH_FLASHATTN) || defined(PADDLE_WITH_COREX)
  // q,k,v [total_*, num_heads, head_dim]
  const auto head_groupnum = qkv.dims()[1];  // nheads/nheads_k + 1 + 1
  DenseTensor q, k, v;
  sliceFlattenView(qkv, &q, 1, 0, head_groupnum - 2);
  sliceFlattenView(qkv, &k, 1, head_groupnum - 2, 1);
  sliceFlattenView(qkv, &v, 1, head_groupnum - 1, 1);
  // DenseTensor dqkv_tmp;
  if (!dqkv) {
    return;
    // dqkv is the only output. No need to compute if no dqkv
    // dqkv_tmp.Resize(qkv.dims());
    // dqkv = &dqkv_tmp;
  }
  dev_ctx.template Alloc<T>(dqkv);
  {
    std::vector<const DenseTensor*> inputs{};
    std::vector<DenseTensor*> outputs{dqkv};
    phi::funcs::ElementwiseKernel<T>(
        dev_ctx, inputs, &outputs, ZeroFunctor<T>());
  }
  DenseTensor dq, dk, dv;
  sliceFlattenView(*dqkv, &dq, 1, 0, head_groupnum - 2);
  sliceFlattenView(*dqkv, &dk, 1, head_groupnum - 2, 1);
  sliceFlattenView(*dqkv, &dv, 1, head_groupnum - 1, 1);
  FlashAttnUnpaddedGradBaseKernel<T>(dev_ctx,
                                     q,
                                     k,
                                     v,
                                     cu_seqlens_q,
                                     cu_seqlens_k,
                                     out,
                                     softmax_lse,
                                     seed_offset,
                                     attn_mask,
                                     dout,
                                     max_seqlen_q,
                                     max_seqlen_k,
                                     scale,
                                     dropout,
                                     causal,
                                     &dq,
                                     &dk,
                                     &dv,
                                     varlen_padded);
#else
  RaiseNotSupportedError();
#endif
}
template <typename T, typename Context>
void FlashAttnGradBaseKernel(
    const Context& dev_ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const paddle::optional<DenseTensor>& startend_row_indices,
    const DenseTensor& dout,
    float dropout,
    bool causal,
    DenseTensor* dq,
    DenseTensor* dk,
    DenseTensor* dv) {
  const auto& dims = q.dims();

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size_og = dout.dims()[3];
  const int64_t head_size = dims[3];
  const int64_t head_size_v = v.dims()[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];
  const int64_t total_q = batch_size * seqlen_q;
  const int64_t total_k = batch_size * seqlen_k;

  bool is_mha = (num_heads == num_heads_k);

  std::initializer_list<int64_t> dk_dv_shape = {
      batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size};

  DenseTensor* kdq = dq;
  DenseTensor dq_tmp;
  if (!dq) {
    dq_tmp.Resize(dims);
    dev_ctx.template Alloc<T>(&dq_tmp);
    kdq = &dq_tmp;
  }

  DenseTensor *kdk = dk, *kdv = dv;
  DenseTensor dk_tmp;
  if (!dk || !is_mha) {
    dk_tmp.Resize({batch_size, seqlen_k, num_heads, head_size});
    dev_ctx.template Alloc<T>(&dk_tmp);
    kdk = &dk_tmp;
  }

  DenseTensor dv_tmp;
  if (!dv || !is_mha) {
    dv_tmp.Resize({batch_size, seqlen_k, num_heads, head_size_v});
    dev_ctx.template Alloc<T>(&dv_tmp);
    kdv = &dv_tmp;
  }

  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      common::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  const float softmax_scale = 1.0f / std::sqrt(head_size);
  const float softmax_unscale = std::sqrt(head_size);

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

  PADDLE_ENFORCE_EQ(out.strides()[out.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "out tensor must have contiguous last dimension"));

  PADDLE_ENFORCE_EQ(dout.strides()[dout.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "dout tensor must have contiguous last dimension"));

  const int64_t* seed_offset_data = seed_offset.data<int64_t>();
  int64_t seed = static_cast<int64_t>(seed_offset_data[0]);
  int64_t offset = static_cast<int64_t>(seed_offset_data[1]);
  PhiloxCudaState philox_state = PhiloxCudaState(seed, offset);

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
    // ixattnbkd bwd
    ixAttnBkdConfigInfo ixAttnbkdInfo;
    ixAttnbkdInfo.stream = dev_ctx.stream();
    ixAttnbkdInfo.softmax_scale = softmax_scale;
    ixAttnbkdInfo.dropout_prob = dropout;
    ixAttnbkdInfo.is_causal = causal;
    ixAttnbkdInfo.causal_mode = 0;
    // ixAttnbkdInfo.is_alibi              = use_alibi;
    // ixAttnbkdInfo.alibi_mode            = alibi_mode;
    ixAttnbkdInfo.return_softmax_lse = false;
    ixAttnbkdInfo.philox_args =
        *(reinterpret_cast<ixAttnBkdPhiloxState*>(&philox_state));
    ixAttnbkdInfo.imp_mode =
        FLAGS_imp_mode ? IXATTNBKD_FATTN_MEM_MODE : IXATTNBKD_FATTN_PERF_MODE;
    ixAttnbkdInfo.is_unpad = false;
    ixAttnbkdInfo.batch = batch_size;
    ixAttnbkdInfo.max_seq_len_src = seqlen_q;
    ixAttnbkdInfo.max_seq_len_trg = seqlen_k;
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

    ixAttnBkdTensorDesc q_desc, k_desc, v_desc, o_desc, m_desc, lse_desc,
        dq_desc, dk_desc, dv_desc, do_desc;
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
    SetIxAttnBkdTensor(&dq_desc, dq, dataType);
    SetIxAttnBkdTensor(&dk_desc, kdq, dataType);
    SetIxAttnBkdTensor(&dv_desc, kdv, dataType);
    SetIxAttnBkdTensor(&do_desc, dout, dataType);

    size_t size_tmpbuf = 0;
    PADDLE_IXATTNBKD_CHECK(
        ixAttnBkdGetFlashAttnBwdBuffSize(batch_size,
                                         num_heads,
                                         num_heads_k,
                                         seqlen_q,
                                         seqlen_k,
                                         head_size,
                                         head_size_v,
                                         total_q,
                                         false,
                                         &size_tmpbuf,
                                         ixAttnbkdInfo.imp_mode));

    auto workspace =
        phi::Empty<uint8_t>(dev_ctx, {static_cast<int64_t>(size_tmpbuf)});
    PADDLE_IXATTNBKD_CHECK(ixAttnBkdFlashAttnBackward(
        ixAttnbkdInfo,
        q_desc,
        k_desc,
        v_desc,
        o_desc,
        m_desc,
        lse_desc,
        dq_desc,
        dk_desc,
        dv_desc,
        do_desc,
        q.data(),
        k.data(),
        v.data(),
        out.data(),
        dout.data(),
        attn_mask.get_ptr() ? (attn_mask.get_ptr())->data() : nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        softmax_lse.data<float>(),
        workspace.data(),
        kdq->data(),
        kdk->data(),
        kdv->data()));

  } else {
    // ixdnn bwd
    cudnnFlashAttnConfigInfo flashAttnInfo;
    flashAttnInfo.softmax_scale = std::sqrt(1.f / head_size);
    flashAttnInfo.dropout_prob = dropout;
    flashAttnInfo.is_causal = causal;
    flashAttnInfo.causal_mode = 0;
    // flashAttnInfo.is_alibi              = use_alibi;
    // flashAttnInfo.alibi_mode            = alibi_mode;
    flashAttnInfo.return_softmax_lse = false;
    flashAttnInfo.philox_args =
        *(reinterpret_cast<cudnnPhiloxCudaState*>(&philox_state));
    flashAttnInfo.imp_mode =
        FLAGS_imp_mode ? CUDNN_FATTN_LEAST_MEM_MODE : CUDNN_FATTN_BALANCE_MODE;
    flashAttnInfo.is_unpad = false;
    flashAttnInfo.batch = batch_size;
    flashAttnInfo.max_seq_len_src = seqlen_q;
    flashAttnInfo.max_seq_len_trg = seqlen_k;
    flashAttnInfo.accuracy_first = accuracy_first;

    int32_t nb_dims = 4;
    std::vector<int32_t> qShape, kShape, vShape, oShape, lseShape, dqShape,
        dkShape, dvShape, doShape;
    std::vector<int32_t> qStride, kStride, vStride, oStride, lseStride,
        dqStride, dkStride, dvStride, doStride;

    cudnnDataType_t dataType;
    if (q.dtype() == phi::DataType::FLOAT16) {
      dataType = CUDNN_DATA_HALF;
    } else if (q.dtype() == phi::DataType::BFLOAT16) {
      dataType = CUDNN_DATA_BFLOAT16;
    } else if (q.dtype() == phi::DataType::FLOAT32) {
      dataType = CUDNN_DATA_FLOAT;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "flash_attn_bwd receive input with dtype == %s",
          phi::DataTypeToString(q.dtype())));
    }

    cudnnHandle_t cudnn;
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnCreate(&cudnn));

    cudnnFlashAttnDescriptor_t flashAttnDesc;
    cudnnTensorDescriptor_t q_desc, k_desc, v_desc, o_desc, m_desc, lse_desc,
        dq_desc, dk_desc, dv_desc, do_desc;

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
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateTensorDescriptor(&dq_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateTensorDescriptor(&dk_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateTensorDescriptor(&dv_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateTensorDescriptor(&do_desc));

    copyDimsAndStrides(q, qShape, qStride);
    copyDimsAndStrides(kdq, dqShape, dqStride);

    copyDimsAndStrides(k, kShape, kStride);
    copyDimsAndStrides(kdk, dkShape, dkStride);

    copyDimsAndStrides(v, vShape, vStride);
    copyDimsAndStrides(kdv, dvShape, dvStride);

    copyDimsAndStrides(out, oShape, oStride);
    copyDimsAndStrides(dout, doShape, doStride);
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
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        dq_desc, dataType, nb_dims, dqShape.data(), dqStride.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        dk_desc, dataType, nb_dims, dkShape.data(), dkStride.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        dv_desc, dataType, nb_dims, dvShape.data(), dvStride.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        do_desc, dataType, nb_dims, doShape.data(), doStride.data()));

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
                                               true,
                                               &size_tmpbuf,
                                               flashAttnInfo.imp_mode));
    auto workspace =
        phi::Empty<uint8_t>(dev_ctx, {static_cast<int64_t>(size_tmpbuf)});

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnFlashAttnBackward(
        cudnn,
        flashAttnDesc,
        flashAttnInfo,
        q_desc,
        k_desc,
        v_desc,
        o_desc,
        attn_mask.get_ptr() ? m_desc : nullptr,
        lse_desc,
        dq_desc,
        dk_desc,
        dv_desc,
        do_desc,
        q.data(),
        k.data(),
        v.data(),
        out.data(),
        dout.data(),
        softmax_lse.data<float>(),
        attn_mask.get_ptr() ? (attn_mask.get_ptr())->data() : nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        workspace.data(),
        kdq->data(),
        kdk->data(),
        kdv->data()));

    auto destroy_tensor_desc = [&](cudnnTensorDescriptor_t desc) {
      if (desc) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnDestroyTensorDescriptor(desc));
      }
    };

    phi::dynload::cudnnDestroyFlashAttnDescriptor(flashAttnDesc);
    destroy_tensor_desc(q_desc);
    destroy_tensor_desc(k_desc);
    destroy_tensor_desc(v_desc);
    destroy_tensor_desc(o_desc);
    destroy_tensor_desc(lse_desc);
    if (attn_mask) {
      destroy_tensor_desc(m_desc);
    }
    destroy_tensor_desc(dq_desc);
    destroy_tensor_desc(dk_desc);
    destroy_tensor_desc(dv_desc);
    destroy_tensor_desc(do_desc);

    flashAttnDesc = nullptr;
    q_desc = nullptr;
    k_desc = nullptr;
    v_desc = nullptr;
    o_desc = nullptr;
    lse_desc = nullptr;
    if (attn_mask) {
      m_desc = nullptr;
    }
    dq_desc = nullptr;
    dk_desc = nullptr;
    dv_desc = nullptr;
    do_desc = nullptr;
  }
  if (!is_mha) {
    if (dk) {
      dk_tmp.Resize(dk_dv_shape);
      if (dk->meta().is_contiguous())
        phi::SumKernel<T, Context>(dev_ctx, dk_tmp, {3}, dk->type(), false, dk);
      else
        kvReduceBatchedForGQA<T, Context>(dev_ctx, dk_tmp, dk);
    }

    if (dv) {
      dv_tmp.Resize(dk_dv_shape);
      if (dv->meta().is_contiguous())
        phi::SumKernel<T, Context>(dev_ctx, dv_tmp, {3}, dv->type(), false, dv);
      else
        kvReduceBatchedForGQA<T, Context>(dev_ctx, dv_tmp, dv);
    }
  }
}

template <typename T, typename Context>
void FlashAttnGradKernel(const Context& dev_ctx,
                         const DenseTensor& q,
                         const DenseTensor& k,
                         const DenseTensor& v,
                         const DenseTensor& out,
                         const DenseTensor& softmax_lse,
                         const DenseTensor& seed_offset,
                         const paddle::optional<DenseTensor>& attn_mask,
                         const DenseTensor& dout,
                         float dropout,
                         bool causal,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
  if (dq) {
    dev_ctx.template Alloc<T>(dq);
  }
  if (dk) {
    dev_ctx.template Alloc<T>(dk);
  }
  if (dv) {
    dev_ctx.template Alloc<T>(dv);
  }
  FlashAttnGradBaseKernel<T, Context>(dev_ctx,
                                      q,
                                      k,
                                      v,
                                      out,
                                      softmax_lse,
                                      seed_offset,
                                      attn_mask,
                                      paddle::none,
                                      dout,
                                      dropout,
                                      causal,
                                      dq,
                                      dk,
                                      dv);
}

template <typename T, typename Context>
void FlashAttnQKVPackedGradKernel(
    const Context& dev_ctx,
    const DenseTensor& qkv,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const DenseTensor& dout,
    float dropout,
    bool causal,
    DenseTensor* dqkv) {
#if defined(PADDLE_WITH_FLASHATTN) || defined(PADDLE_WITH_COREX)
  // qkv [batchsize, seqlen, nheads/nheads_k+2, nheads_k, head_dim]
  const auto head_groupnum = qkv.dims()[2];  // nheads/nheads_k + 1 + 1
  DenseTensor q, k, v;
  sliceFlattenView(qkv, &q, 2, 0, head_groupnum - 2);
  sliceFlattenView(qkv, &k, 2, head_groupnum - 2, 1);
  sliceFlattenView(qkv, &v, 2, head_groupnum - 1, 1);
  // DenseTensor dqkv_tmp;
  if (!dqkv) {
    return;
    // dqkv is the only output. No need to compute if no dqkv
    // dqkv_tmp.Resize(qkv.dims());
    // dqkv = &dqkv_tmp;
  }
  dev_ctx.template Alloc<T>(dqkv);
  DenseTensor dq, dk, dv;
  sliceFlattenView(*dqkv, &dq, 2, 0, head_groupnum - 2);
  sliceFlattenView(*dqkv, &dk, 2, head_groupnum - 2, 1);
  sliceFlattenView(*dqkv, &dv, 2, head_groupnum - 1, 1);
  FlashAttnGradBaseKernel<T, Context>(dev_ctx,
                                      q,
                                      k,
                                      v,
                                      out,
                                      softmax_lse,
                                      seed_offset,
                                      attn_mask,
                                      paddle::none,
                                      dout,
                                      dropout,
                                      causal,
                                      &dq,
                                      &dk,
                                      &dv);
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashMaskGradKernel(const Context& dev_ctx,
                         const DenseTensor& q,
                         const DenseTensor& k,
                         const DenseTensor& v,
                         const DenseTensor& startend_row_indices,
                         const DenseTensor& out,
                         const DenseTensor& softmax_lse,
                         const DenseTensor& seed_offset,
                         const DenseTensor& dout,
                         float dropout,
                         bool causal,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
#ifdef PADDLE_WITH_COREX
  RaiseNotSupportedError();
#else
  if (dq) {
    dev_ctx.template Alloc<T>(dq);
  }
  if (dk) {
    dev_ctx.template Alloc<T>(dk);
  }
  if (dv) {
    dev_ctx.template Alloc<T>(dv);
  }
  FlashAttnGradBaseKernel<T, Context>(dev_ctx,
                                      q,
                                      k,
                                      v,
                                      out,
                                      softmax_lse,
                                      seed_offset,
                                      paddle::none,
                                      startend_row_indices,
                                      dout,
                                      dropout,
                                      causal,
                                      dq,
                                      dk,
                                      dv);
#endif
}
}  // namespace phi

PD_REGISTER_PLUGIN_KERNEL(flash_attn_unpadded_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnUnpaddedGradKernel,
#ifdef PADDLE_WITH_COREX
                          float,
#endif
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(7).SetBackend(phi::Backend::CPU);  // seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flash_attn_varlen_qkvpacked_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnVarlenQKVPackedGradKernel,
#ifdef PADDLE_WITH_COREX
                          float,
#endif
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(phi::Backend::CPU);  // seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flash_attn_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnGradKernel,
#ifdef PADDLE_WITH_COREX
                          float,
#endif
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(phi::Backend::CPU);  // seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flash_attn_qkvpacked_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnQKVPackedGradKernel,
#ifdef PADDLE_WITH_COREX
                          float,
#endif
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(3).SetBackend(phi::Backend::CPU);  // seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flashmask_attention_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashMaskGradKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(6).SetBackend(phi::Backend::CPU);  // seed_offset
}
