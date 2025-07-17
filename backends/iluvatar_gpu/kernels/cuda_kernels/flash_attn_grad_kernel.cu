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
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/pad_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"
COMMON_DECLARE_int32(ixdnn_imp_mode);
COMMON_DECLARE_int32(ixdnn_causal_mode);

COMMON_DECLARE_bool(cudnn_deterministic);

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
                    phi::errors::InvalidArgument(
                        "FlashAttention only support headdim <= 256"));
  PADDLE_ENFORCE_EQ(headdim % 32,
                    0,
                    phi::errors::InvalidArgument(
                        "FlashAttention only support headdim %% 32 == 0"));
  PADDLE_ENFORCE_NE(
      headdim, 0, phi::errors::InvalidArgument("Headdim can't be zero"));
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
static void kvReduceForGQA(const Context& ctx,
                           const DenseTensor& dk_tmp,
                           DenseTensor* dk) {
  PADDLE_ENFORCE_EQ(
      dk->strides()[2],
      1,
      phi::errors::InvalidArgument("headdim dimention must be contiguous"));
  PADDLE_ENFORCE_EQ(
      dk_tmp.strides()[3],
      1,
      phi::errors::InvalidArgument("headdim dimention must be contiguous"));
  const int64_t reduceDimSize = dk_tmp.dims()[2];
  const size_t blockNum =
      std::min((static_cast<int64_t>(dk_tmp.dims()[0] + 31) / 32),
               static_cast<int64_t>(1024l));
  const dim3 threadNum{32, 4, 1};
  auto sumkernel = selectSumkernel<T>(dk_tmp.dims()[3]);
  sumkernel<<<blockNum, threadNum, 0, ctx.stream()>>>(
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
static void kvReduceBatchedForGQA(const Context& ctx,
                                  const DenseTensor& dk_tmp,
                                  DenseTensor* dk) {
  PADDLE_ENFORCE_EQ(
      dk->strides()[3],
      1,
      phi::errors::InvalidArgument("headdim dimention must be contiguous"));
  PADDLE_ENFORCE_EQ(
      dk_tmp.strides()[4],
      1,
      phi::errors::InvalidArgument("headdim dimention must be contiguous"));
  PADDLE_ENFORCE_EQ(
      dk->strides()[0],
      dk->strides()[1] * dk->dims()[1],
      phi::errors::InvalidArgument("batchsize dimention must be contiguous"));
  PADDLE_ENFORCE_EQ(
      dk_tmp.strides()[0],
      dk_tmp.strides()[1] * dk_tmp.dims()[1],
      phi::errors::InvalidArgument("batchsize dimention must be contiguous"));
  const int64_t reduceDimSize = dk_tmp.dims()[3];
  const size_t blockNum = std::min(
      (static_cast<int64_t>(dk_tmp.dims()[0] * dk_tmp.dims()[1] + 31) / 32),
      static_cast<int64_t>(1024l));
  const dim3 threadNum{32, 4, 1};
  auto sumkernel = selectSumkernel<T>(dk_tmp.dims()[4]);
  // here implicitly flat [batch,seqlen], and require batch dim to be contiguous
  sumkernel<<<blockNum, threadNum, 0, ctx.stream()>>>(
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
    const Context& ctx,
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
  // q,k,v [total_*, num_heads, head_dim]
  auto dims = q.dims();

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size_og = dout.dims()[2];
  const int64_t head_size = dims[2];
  const int64_t total_k = k.dims()[0];
  const int64_t num_heads_k = k.dims()[1];
  int64_t max_seqlen_q = max_seqlen_q_.to<int64_t>();
  int64_t max_seqlen_k = max_seqlen_k_.to<int64_t>();

  bool is_mha = (num_heads == num_heads_k);

  int64_t total_q = dims[0];
  bool is_unpad = true;
  const int64_t head_size_rounded = head_size + 32 - head_size % 32;

  DenseTensor q_padded, k_padded, v_padded, out_padded, dout_padded;
  q_padded = q;
  k_padded = k;
  v_padded = v;
  out_padded = out;
  dout_padded = dout;

  DenseTensor* kdq = dq;
  DenseTensor dq_tmp;
  if (!dq) {
    dq_tmp.Resize(q_padded.dims());
    ctx.template Alloc<T>(&dq_tmp);
    kdq = &dq_tmp;
  }

  DenseTensor *kdk = dk, *kdv = dv;
  DenseTensor dk_tmp, dv_tmp;

  if (!dk || !is_mha) {
    if (is_unpad)
      dk_tmp.Resize({total_k, num_heads, head_size});
    else
      dk_tmp.Resize({batch_size, max_seqlen_k, num_heads, head_size});
    ctx.template Alloc<T>(&dk_tmp);
    kdk = &dk_tmp;
  }

  if (!dv || !is_mha) {
    if (is_unpad)
      dv_tmp.Resize({total_k, num_heads, head_size});
    else
      dv_tmp.Resize({batch_size, max_seqlen_k, num_heads, head_size});
    ctx.template Alloc<T>(&dv_tmp);
    kdv = &dv_tmp;
  }

  const cudaStream_t stream = ctx.stream();

  int num_splits = get_num_split();

  // TODO(umiswing): add shape check
  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      phi::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  // ixdnn
  const int64_t* seed_offset_data = seed_offset.data<int64_t>();
  int64_t seed = static_cast<int64_t>(seed_offset_data[0]);
  int64_t offset = static_cast<int64_t>(seed_offset_data[1]);
  PhiloxCudaState philox_state = PhiloxCudaState(seed, offset);

  cudnnFlashAttnConfigInfo flashAttnInfo;
  flashAttnInfo.softmax_scale = std::sqrt(1.f / head_size);
  flashAttnInfo.dropout_prob = dropout;
  flashAttnInfo.is_causal = causal;
  flashAttnInfo.causal_mode = FLAGS_ixdnn_causal_mode;
  // flashAttnInfo.is_alibi              = use_alibi;
  // flashAttnInfo.alibi_mode            = alibi_mode;
  flashAttnInfo.return_softmax_lse = false;
  flashAttnInfo.philox_args =
      *(reinterpret_cast<cudnnPhiloxCudaState*>(&philox_state));
  flashAttnInfo.imp_mode = FLAGS_ixdnn_imp_mode ? CUDNN_FATTN_LEAST_MEM_MODE
                                                : CUDNN_FATTN_BALANCE_MODE;
  flashAttnInfo.is_unpad = is_unpad;
  flashAttnInfo.batch = batch_size;
  flashAttnInfo.max_seq_len_src = max_seqlen_q;
  flashAttnInfo.max_seq_len_trg = max_seqlen_k;

  int32_t nb_dims = is_unpad ? 3 : 4;
  std::vector<int32_t> qShape, kShape, vShape, oShape, lseShape, dqShape,
      dkShape, dvShape, doShape;
  std::vector<int32_t> qStride, kStride, vStride, oStride, lseStride, dqStride,
      dkStride, dvStride, doStride;

  cudnnDataType_t dataType;
  if (q.dtype() == phi::DataType::FLOAT16) {
    dataType = CUDNN_DATA_HALF;
  } else if (q.dtype() == phi::DataType::BFLOAT16) {
    dataType = CUDNN_DATA_BFLOAT16;
  }

  cudnnHandle_t cudnn;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnCreate(&cudnn));

  cudnnFlashAttnDescriptor_t flashAttnDesc;
  cudnnTensorDescriptor_t q_desc, k_desc, v_desc, o_desc, lse_desc, dq_desc,
      dk_desc, dv_desc, do_desc;

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

  DenseTensor out_s, q_, k_, v_, dout_;
  out_s.ShareDataWith(out).Resize({total_q, num_heads, head_size});
  q_ = q_padded;
  k_ = k_padded;
  v_ = v_padded;
  dout_ = dout_padded;

  if (!is_unpad) {
    q_.Resize({batch_size, max_seqlen_q, num_heads, head_size});
    k_.Resize({batch_size, max_seqlen_k, num_heads_k, head_size});
    v_.Resize({batch_size, max_seqlen_k, num_heads_k, head_size});
    out_s.Resize({batch_size, max_seqlen_q, num_heads, head_size});
    dq->Resize({batch_size, max_seqlen_q, num_heads, head_size});
    dk->Resize({batch_size, max_seqlen_k, num_heads_k, head_size});
    dv->Resize({batch_size, max_seqlen_k, num_heads_k, head_size});
    dout_.Resize({batch_size, max_seqlen_q, num_heads, head_size});
  }

  copyDimsAndStrides(q_, qShape, qStride);
  copyDimsAndStrides(kdq, dqShape, dqStride);

  copyDimsAndStrides(k_, kShape, kStride);
  copyDimsAndStrides(kdk, dkShape, dkStride);

  copyDimsAndStrides(v_, vShape, vStride);
  copyDimsAndStrides(kdv, dvShape, dvStride);

  copyDimsAndStrides(out_s, oShape, oStride);
  copyDimsAndStrides(dout_, doShape, doStride);
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
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      dq_desc, dataType, nb_dims, dqShape.data(), dqStride.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      dk_desc, dataType, nb_dims, dkShape.data(), dkStride.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      dv_desc, dataType, nb_dims, dvShape.data(), dvStride.data()));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
      do_desc, dataType, nb_dims, doShape.data(), doStride.data()));

  size_t size_tmpbuf = 0;
  void* d_wkSpace = nullptr;

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
                                             true,
                                             &size_tmpbuf,
                                             flashAttnInfo.imp_mode));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_wkSpace), size_tmpbuf));
  CUDA_CHECK(cudaMemset(d_wkSpace, 0xff, size_tmpbuf));

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnFlashAttnBackward(
      cudnn,
      flashAttnDesc,
      flashAttnInfo,
      q_desc,
      k_desc,
      v_desc,
      o_desc,
      nullptr,
      lse_desc,
      dq_desc,
      dk_desc,
      dv_desc,
      do_desc,
      q_padded.data(),
      k_padded.data(),
      v_padded.data(),
      out_padded.data(),
      dout_padded.data(),
      softmax_lse.data<float>(),
      nullptr,
      is_unpad ? cu_seqlens_q.data<int>() : nullptr,
      is_unpad ? cu_seqlens_k.data<int>() : nullptr,
      nullptr,  // reinterpret_cast<int *>(d_loWinIdx.data_ptr()),
      nullptr,  // reinterpret_cast<int *>(d_hiWinIdx.data_ptr()),
      nullptr,
      d_wkSpace,
      kdq->data(),
      kdk->data(),
      kdv->data()));

  if (!is_mha) {
    int reduce_axis = is_unpad ? 2 : 3;
    if (dk) {
      if (is_unpad)
        dk_tmp.Resize(
            {total_k, num_heads_k, num_heads / num_heads_k, head_size});
      else
        dk_tmp.Resize({batch_size,
                       max_seqlen_k,
                       num_heads_k,
                       num_heads / num_heads_k,
                       head_size});
      if (dk->meta().is_contiguous())
        phi::SumKernel<T, Context>(
            ctx, dk_tmp, {reduce_axis}, dk->type(), false, dk);
      else
        kvReduceForGQA<T, Context>(ctx, dk_tmp, dk);
    }
    if (dv) {
      if (is_unpad)
        dv_tmp.Resize(
            {total_k, num_heads_k, num_heads / num_heads_k, head_size});
      else
        dv_tmp.Resize({batch_size,
                       max_seqlen_k,
                       num_heads_k,
                       num_heads / num_heads_k,
                       head_size});
      if (dv->meta().is_contiguous())
        phi::SumKernel<T, Context>(
            ctx, dv_tmp, {reduce_axis}, dv->type(), false, dv);
      else
        kvReduceForGQA<T, Context>(ctx, dv_tmp, dv);
    }
  }

  phi::dynload::cudnnDestroyFlashAttnDescriptor(flashAttnDesc);
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(q_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(k_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(v_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(o_desc));
  CUDA_CHECK(cudaFree(d_wkSpace));
  flashAttnDesc = nullptr;
  q_desc = nullptr;
  k_desc = nullptr;
  v_desc = nullptr;
  o_desc = nullptr;
}

template <typename T, typename Context>
void FlashAttnUnpaddedGradKernel(const Context& ctx,
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
  if (dq) {
    ctx.template Alloc<T>(dq);
  }
  if (dk) {
    ctx.template Alloc<T>(dk);
  }
  if (dv) {
    ctx.template Alloc<T>(dv);
  }
  FlashAttnUnpaddedGradBaseKernel<T>(ctx,
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
template <typename OutT>
struct ZeroFunctor {
  __device__ __forceinline__ OutT operator()() const {
    return static_cast<OutT>(0);
  }
};

template <typename T, typename Context>
void FlashAttnGradBaseKernel(
    const Context& ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const paddle::optional<DenseTensor>& attn_mask_start_row_indices,
    const DenseTensor& dout,
    float dropout,
    bool causal,
    int attn_mask_start_row,
    DenseTensor* dq,
    DenseTensor* dk,
    DenseTensor* dv) {
  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size_og = dout.dims()[3];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];

  bool is_mha = (num_heads == num_heads_k);
  const int64_t head_size_rounded = head_size + 32 - head_size % 32;

  DenseTensor q_padded, k_padded, v_padded, out_padded, dout_padded;
  q_padded = q;
  k_padded = k;
  v_padded = v;
  out_padded = out;
  dout_padded = dout;

  DenseTensor* kdq = dq;
  DenseTensor dq_tmp;

  if (!dq) {
    dq_tmp.Resize(q_padded.dims());
    ctx.template Alloc<T>(&dq_tmp);
    kdq = &dq_tmp;
  }

  DenseTensor *kdk = dk, *kdv = dv;
  DenseTensor dk_tmp;
  if (!dk || !is_mha) {
    if (!is_mha) {
      dk_tmp.Resize({batch_size, seqlen_k, num_heads, head_size_og});
    } else {
      dk_tmp.Resize(k_padded.dims());
    }
    ctx.template Alloc<T>(&dk_tmp);
    kdk = &dk_tmp;
  }

  DenseTensor dv_tmp;
  if (!dv || !is_mha) {
    if (!is_mha) {
      dv_tmp.Resize({batch_size, seqlen_k, num_heads, head_size_og});
    } else {
      dv_tmp.Resize(v_padded.dims());
    }
    ctx.template Alloc<T>(&dv_tmp);
    kdv = &dv_tmp;
  }

  const cudaStream_t stream = ctx.stream();

  // TODO(umiswing): add shape check
  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      phi::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  const float softmax_scale = 1.0f / std::sqrt(head_size);
  const float softmax_unscale = std::sqrt(head_size);

  // ixdnn
  int64_t total_q = batch_size * seqlen_q;

  const int64_t* seed_offset_data = seed_offset.data<int64_t>();
  int64_t seed = static_cast<int64_t>(seed_offset_data[0]);
  int64_t offset = static_cast<int64_t>(seed_offset_data[1]);

  PhiloxCudaState philox_state = PhiloxCudaState(seed, offset);

  cudnnFlashAttnConfigInfo flashAttnInfo;
  flashAttnInfo.softmax_scale = softmax_scale;
  flashAttnInfo.dropout_prob = dropout;
  flashAttnInfo.is_causal = causal;
  flashAttnInfo.causal_mode = FLAGS_ixdnn_causal_mode;
  // flashAttnInfo.is_alibi              = use_alibi;
  // flashAttnInfo.alibi_mode            = alibi_mode;
  flashAttnInfo.return_softmax_lse = false;
  flashAttnInfo.philox_args =
      *(reinterpret_cast<cudnnPhiloxCudaState*>(&philox_state));
  flashAttnInfo.imp_mode = FLAGS_ixdnn_imp_mode ? CUDNN_FATTN_LEAST_MEM_MODE
                                                : CUDNN_FATTN_BALANCE_MODE;
  flashAttnInfo.is_unpad = false;
  flashAttnInfo.batch = batch_size;
  flashAttnInfo.max_seq_len_src = seqlen_q;
  flashAttnInfo.max_seq_len_trg = seqlen_k;

  int32_t nb_dims = 4;
  std::vector<int32_t> qShape, kShape, vShape, oShape, lseShape, dqShape,
      dkShape, dvShape, doShape;
  std::vector<int32_t> qStride, kStride, vStride, oStride, lseStride, dqStride,
      dkStride, dvStride, doStride;

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

  copyDimsAndStrides(q_padded, qShape, qStride);
  copyDimsAndStrides(kdq, dqShape, dqStride);

  copyDimsAndStrides(k_padded, kShape, kStride);
  copyDimsAndStrides(kdk, dkShape, dkStride);

  copyDimsAndStrides(v_padded, vShape, vStride);
  copyDimsAndStrides(kdv, dvShape, dvStride);

  copyDimsAndStrides(out_padded, oShape, oStride);
  copyDimsAndStrides(dout_padded, doShape, doStride);
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
  void* d_wkSpace = nullptr;

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

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_wkSpace), size_tmpbuf));
  CUDA_CHECK(cudaMemset(d_wkSpace, 0xff, size_tmpbuf));

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
      q_padded.data(),
      k_padded.data(),
      v_padded.data(),
      out_padded.data(),
      dout_padded.data(),
      softmax_lse.data<float>(),
      attn_mask.get_ptr() ? (attn_mask.get_ptr())->data() : nullptr,
      nullptr,
      nullptr,
      nullptr,  // reinterpret_cast<int *>(d_loWinIdx.data_ptr()),
      nullptr,  // reinterpret_cast<int *>(d_hiWinIdx.data_ptr()),
      nullptr,
      d_wkSpace,
      kdq->data(),
      kdk->data(),
      kdv->data()));

  if (!is_mha) {
    if (dk) {
      dk_tmp.Resize({batch_size,
                     seqlen_k,
                     num_heads_k,
                     num_heads / num_heads_k,
                     head_size_og});
      if (dk->meta().is_contiguous())
        phi::SumKernel<T, Context>(ctx, dk_tmp, {3}, dk->type(), false, dk);
      else
        kvReduceBatchedForGQA<T, Context>(ctx, dk_tmp, dk);
    }

    if (dv) {
      dv_tmp.Resize({batch_size,
                     seqlen_k,
                     num_heads_k,
                     num_heads / num_heads_k,
                     head_size_og});
      if (dv->meta().is_contiguous())
        phi::SumKernel<T, Context>(ctx, dv_tmp, {3}, dv->type(), false, dv);
      else
        kvReduceBatchedForGQA<T, Context>(ctx, dv_tmp, dv);
    }
  }

  phi::dynload::cudnnDestroyFlashAttnDescriptor(flashAttnDesc);
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(q_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(k_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(v_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(o_desc));
  CUDA_CHECK(cudaFree(d_wkSpace));
  flashAttnDesc = nullptr;
  q_desc = nullptr;
  k_desc = nullptr;
  v_desc = nullptr;
  o_desc = nullptr;
}

template <typename T, typename Context>
void FlashAttnGradKernel(const Context& ctx,
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
    ctx.template Alloc<T>(dq);
  }
  if (dk) {
    ctx.template Alloc<T>(dk);
  }
  if (dv) {
    ctx.template Alloc<T>(dv);
  }
  FlashAttnGradBaseKernel<T, Context>(ctx,
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
                                      0,
                                      dq,
                                      dk,
                                      dv);
}
}  // namespace phi

PD_REGISTER_PLUGIN_KERNEL(flash_attn_unpadded_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnUnpaddedGradKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}

PD_REGISTER_PLUGIN_KERNEL(flash_attn_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlashAttnGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}
