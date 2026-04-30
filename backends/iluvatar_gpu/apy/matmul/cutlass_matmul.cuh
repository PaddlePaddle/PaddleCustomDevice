// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include "cute/atom/mma_traits.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include "cutlass/detail/layout.hpp"

#include "cute/atom/copy_traits_ix11_sme.hpp"
#include "cute/atom/copy_traits_ix11.hpp"
#include "cute/atom/copy_atom.hpp"

#include "cutlass/arch/config.h"
#include "cutlass/epilogue/dispatch_policy.hpp"

#include "cutlass_patch/batched_matrix_coord.h"
#include "cutlass_patch/epilogue/collective/ix11_epilogue_vectorized_perwarp_variadic.hpp"
#include "cutlass_patch/epilogue/thread/linear_combination_variadic.h"

#include "default_config_id.h"
#include "params.h"

#define CHECK_CUTLASS(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

namespace ap {
using bfloat16 = nv_bfloat16;

template <typename T, int N>
using Array = cutlass::Array<T, N>;

using MatrixCoord = cutlass::BatchedMatrixCoord;

// Convert CUDA data type to cutlass data type
template <typename T>
struct CutlassDataType {
  using Type = T;
};

template <>
struct CutlassDataType<half> {
  using Type = cutlass::half_t;
};

template <>
struct CutlassDataType<__nv_bfloat16> {
  using Type = cutlass::bfloat16_t;
};

// Convert to cutlass layout
template <bool Transposed>
struct MatrixLayout {
  using Type = cutlass::layout::RowMajor;
};

template <>
struct MatrixLayout<true> {
  using Type = cutlass::layout::ColumnMajor;
};

static cutlass::gemm::GemmUniversalMode GetGemmMode(int batch_count) {
  return batch_count > 1 ? cutlass::gemm::GemmUniversalMode::kBatched
                         : cutlass::gemm::GemmUniversalMode::kGemm;
}

template <typename ElementT,
          typename ElementComputeT,
          template <typename T>
          class VariadicFunctor,
          int AlignA = 64 / sizeof(ElementT),
          int AlignB = 64 / sizeof(ElementT),
          int ConfigId = 0>
void MatmulAddVariadic(
    const GemmEpilogueParams &params,
    const typename VariadicFunctor<ElementComputeT>::Arguments &variadic_args) {
  
  using namespace cute;

  using ElementAccumulator =
      typename CutlassDataType<ElementComputeT>::Type;  // <- data type of
                                                        // accumulator
  using ElementComputeEpilogue =
      ElementAccumulator;  // <- data type of epilogue operations
  using ElementA =
      typename CutlassDataType<ElementT>::Type;  // <- data type of elements in
                                                 // input matrix A
  using ElementB =
      typename CutlassDataType<ElementT>::Type;  // <- data type of elements in
                                                 // input matrix B
  using ElementC = ElementA;
  using ElementD = ElementA;
  using ElementOutput =
      typename CutlassDataType<ElementT>::Type;  // <- data type of elements in
                                                 // output matrix D

  constexpr int AlignC = AlignB;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using EpilogueThreadOp = cutlass::epilogue::thread::LinearCombinationVariadic<
      VariadicFunctor, ElementD, 1, ElementAccumulator, ElementAccumulator>;

  // Epilogue --------------------------------
  // Params
  // Tile of C/D
  using EpilogueTile = Tile<
    Layout<Shape<_16,_4>,Stride<_1,_64>>,
    Layout<Shape<_32,_4>,Stride<_1,_64>>
  >;

  // Layout of smem
  using SmemLayout = 
    ComposedLayout<
      Swizzle<1,1,6>,
      _0,
      Layout<
        Shape <Shape < _4,_2,   _2,  _4>,Shape <_16,  _2,   _4>>,
        Stride<Stride<_32,_1,_4096,_256>,Stride< _2,_128,_1024>>
      >
    >;

  // G2R
  using EpiG2RAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, ElementC>;

  // R2S
  using EpiR2SAtom = Copy_Atom<UniversalCopy<uint32_t>, ElementD>;

  // S2R
  using EpiS2RAtom = EpiR2SAtom;
  using TiledCopyS2R =
    TiledCopy<
      EpiS2RAtom,
      Layout<
        Shape <Shape < _16,_4, _4,   _4>,Shape <_2, _2,_2>>,
        Stride<Stride<_128,_1,_16,_2048>,Stride<_4,_64,_8>>>,
      decltype(product_each(shape(EpilogueTile{})))
    >;

  // R2R
  using CopyAtomR2R = Copy_Atom<U16_2x2Trans,ElementD>;

  // R2G
  using EpiR2GAtom = Copy_Atom<UniversalCopy<uint32_t>, ElementD>;

  // Epilogue Collective
  using CollectiveEpilogue = cutlass::epilogue::collective::EpilogueVariadic<
      // cutlass::detail::TagToStrideC_t<LayoutC>,
      cutlass::detail::TagToStrideC_t<LayoutD>,
      EpilogueThreadOp,
      EpilogueTile,
      SmemLayout,
      EpiG2RAtom,
      EpiR2SAtom,
      TiledCopyS2R,
      CopyAtomR2R,
      EpiR2GAtom,
      cutlass::epilogue::EpilogueSimtVectorized>;

  using CtaTiler = Shape<_256,_256,_32>;
  using TiledMma = 
    TiledMMA<
      MMA_Atom<IX11_16x16x16_F32F16F16F32_TT>,
      Layout<Shape<_4,_4,_1>>,
      Tile<
        Layout<Shape<_16,_4,_4>, Stride<_1,_64,_16>>,
        Layout<Shape<_16,_4,_4>, Stride<_1,_64,_16>>,
        _16
      >
    >;

  using CopyA_Op = IX11_SME_I_16x512b<IX11::SMESwizzle::Row16b, IX11::CacheOP::CacheAll>;
  using CopyA_Atom = Copy_Atom<Copy_Traits<CopyA_Op>, ElementA>;
  using CopyA = decltype(make_tiled_copy(CopyA_Atom{0}, Layout<Shape<_16,_1>>{}, Layout<Shape<_16,_32>,Stride<_1,_16>>{}));
  using SmemLayoutAtomA = IX11::Layout_SME_I_16x512b_Atom<IX11::SMESwizzle::Row16b, ElementA, IX11::Major::K>;
  using SmemCopyA = Copy_Atom<UniversalCopy<uint32_t>, ElementA>;

  using CopyB_Op = IX11_SME_I_16x512b<IX11::SMESwizzle::Col, IX11::CacheOP::CacheAll>;
  using CopyB_Atom = Copy_Atom<Copy_Traits<CopyB_Op>, ElementB>;
  using CopyB = decltype(make_tiled_copy(CopyB_Atom{0}, Layout<Shape<_16,_1>>{}, Layout<Shape<_16,_32>,Stride<_1,_16>>{}));
  using SmemLayoutAtomB = IX11::Layout_SME_I_16x512b_Atom<IX11::SMESwizzle::Col, ElementB, IX11::Major::K>;
  using SmemCopyB = Copy_Atom<UniversalCopy<uint32_t>, ElementB>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopIx11SmeUnpredicated<2>,
    CtaTiler,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    CopyA, SmemLayoutAtomA, SmemCopyA, cute::identity,
    CopyB, SmemLayoutAtomB, SmemCopyB, cute::identity
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  ProblemShapeType problem_shape{params.m, params.n, params.k, params.batch_count};
  
  const ElementA *input =
      reinterpret_cast<const ElementA *>(params.input);
  const ElementB *weight =
      reinterpret_cast<const ElementB *>(params.weight);
  ElementOutput *output = reinterpret_cast<ElementOutput *>(params.output);
  
  cutlass::KernelHardwareInfo hw_info;
  // TODO
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename EpilogueThreadOp::Params epilogue_op_args;
  epilogue_op_args.variadic_args = variadic_args;

  int m = params.m, n = params.n, k = params.k, l = params.batch_count;
  
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_shape,
    {input, stride_A, weight, stride_B},
    {epilogue_op_args, output, stride_D},
    hw_info
  };

  Gemm device_gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cudaStream_t* stream_ptr = reinterpret_cast<cudaStream_t*>(params.stream_ptr);

  CHECK_CUTLASS(device_gemm.can_implement(arguments));
  CHECK_CUTLASS(device_gemm.initialize(arguments, workspace.get(), *stream_ptr));
  CHECK_CUTLASS(device_gemm(*stream_ptr));

}

}  // namespace ap
