/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  // class StrideC,
  class StrideD,
  class ThreadEpilogueOp,
  class EpilogueTile,
  class SmemLayout,
  class CopyAtomG2R_,
  class CopyAtomR2S,
  class TiledCopyS2R,
  class CopyAtomR2R,
  class CopyAtomR2G,
  class EpilogueScheduleType = EpilogueSimtVectorized,
  class Enable = void
>
class EpilogueVariadic {
  static_assert(cute::is_same_v<EpilogueScheduleType, EpilogueSimtVectorized> ||
                cute::is_same_v<EpilogueScheduleType, EpiloguePtrArraySimtVectorized>,
                "Could not find an epilogue specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Iluvatar ix11/ix30 Epilogue Vectorized
/// 1. Load C(optional), compute D = alpha * Acc + beta * C, 
///   then transform to ElementD(8/16-bit typically)
/// 2. Copy D from Register to Smem
/// 3. Load D from Smem to Register
/// 4. Register-to-Register transform
/// 5. Store D from Register to Gmem
///
/// Support alpha and beta params, bias not supported yet.
template <
  // class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class EpilogueTile_,
  class SmemLayout_,
  class CopyAtomG2R_,
  class CopyAtomR2S_,
  class TiledCopyS2R_,
  class CopyAtomR2R_,
  class CopyAtomR2G_,
  class EpilogueScheduleType_
>
class EpilogueVariadic<
        // StrideC_,
        StrideD_,
        ThreadEpilogueOp_,
        EpilogueTile_,
        SmemLayout_,
        CopyAtomG2R_,
        CopyAtomR2S_,
        TiledCopyS2R_,
        CopyAtomR2R_,
        CopyAtomR2G_,
        EpilogueScheduleType_,
        cute::enable_if_t<
          cute::is_same_v<EpilogueScheduleType_, EpilogueSimtVectorized>
        >
      > {
public:
  //
  // Type Aliases
  //
  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementD = typename ThreadEpilogueOp::ElementOutput;
  using StrideD = StrideD_;
  using ElementC = ElementD; // for GemmUniversal
  using StrideC = StrideD;
  using ElementBias = typename detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::type;
  using EpilogueTile = EpilogueTile_;
  using SmemLayout   = SmemLayout_;
  using CopyAtomG2R = CopyAtomG2R_;
  using CopyAtomR2S  = CopyAtomR2S_;
  using TiledCopyS2R = TiledCopyS2R_;
  using CopyAtomR2R  = CopyAtomR2R_;
  using CopyAtomR2G  = CopyAtomR2G_;

  // using GmemTiledCopyC = void;
  using GmemTiledCopyD = CopyAtomR2G;

  static constexpr bool IsEpilogueBiasSupported = detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::value;
  using StrideBias = cute::conditional_t<detail::is_m_major<StrideD>(), Stride<_1,_0,int64_t>, Stride<_0,_1,int64_t>>;

  // static_assert(cute::rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(cute::rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage
  {
    cute::array_aligned<ElementD, cute::cosize_v<SmemLayout>> smem_epilogue;
  };

  struct Arguments {
    typename ThreadEpilogueOp::Params epilogue_op{};
    // using StrideBias = decltype(thread.dBias);
    // ElementC const* ptr_C = nullptr;
    // StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  template<class ThreadEpiOp>
  struct ParamsType {
    typename ThreadEpiOp::Params epilogue_op{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

  using Params = ParamsType<ThreadEpilogueOp>;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& _,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {

    return {
        args.epilogue_op,
        args.ptr_D,
        args.dD,
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  EpilogueVariadic(Params const& params_)
      : params(params_), epilogue_op(params_.epilogue_op) { }

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    return epilogue_op.is_source_needed();
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine,FrgLayout> const& accumulators,                   // (MMA,MMA_M,MMA_N)
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf) {
    using namespace cute;
    using X = Underscore;

    static_assert(cute::rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(cute::rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(cute::rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    static_assert(rank(EpilogueTile{}) == 2, "Rank of EpilogueTile should be 2");
    static_assert(rank(SmemLayout{}) == 2, "Rank of SmemLayout should be 2");
    CUTE_STATIC_ASSERT(size(SmemLayout{}) == size<0>(EpilogueTile{}) * size<1>(EpilogueTile{}));

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    // Represent the full output tensor
    // Tensor mC_mnl = make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M,N,L), params.dC);             //             (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N,L), params.dD);             //             (m,n,l)
    // Tensor mBias_mnl = make_tensor(make_gmem_ptr(params.ptr_Bias), make_shape(M,N,L), params.dBias);    //             (m,n,l)

    // Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});             // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});             // (BLK_M,BLK_N,m,n,l)
    // Tensor gBias_mnl = local_tile(mBias_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});       // (BLK_M,BLK_N,m,n,l)

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    // Tensor gC = gC_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    // Tensor gBias = gBias_mnl(_,_,m_coord,n_coord,l_coord);                                             // (BLK_M,BLK_N)

    // Construct a tensor in SMEM that we can partition for rearranging data
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Common part (with sm70 epilogue) end--------------------------------
    // init register transform
    // TV -> MN(logical)
    auto layout_gd_tv = tiled_mma.thrfrg_C(make_layout(gD.shape()));

    // G2R / R2S -------------------------------------
    // In G2R / R2S tiling, we assemble the final V->MN layout out from given epilogue tile and mma layout

    // epilogue tile of logical C
    auto tE_D  = flat_divide(make_layout(product_each(gD.shape())), EpilogueTile{});
    
    using EpilogueTileSize = decltype(product_each(shape(EpilogueTile{})));

    Tensor sAcc = make_tensor(make_smem_ptr(storage.smem_epilogue.data()), SmemLayout{});

    auto r2s_mn_tiler_tv = take<0,2>(right_inverse(layout_gd_tv).compose(tE_D)); // epilogue logical mn->tv
    // get tiler of TV from tiler of MN
    auto r2s_tv_tiler_shape = product_each(layout_gd_tv.shape());
    auto r2s_tv_zeros = repeat_like(r2s_tv_tiler_shape, Int<0>{}); // T and V
    auto r2s_tv_tiler = cute::transform(make_seq<rank(r2s_tv_tiler_shape)>{}, [&](auto i) {
      auto tiler_origin = filter(composition(make_layout(r2s_tv_tiler_shape, replace<i>(r2s_tv_zeros, Int<1>{})), r2s_mn_tiler_tv));
      auto complemented = complement(tiler_origin, get<i>(r2s_tv_tiler_shape));
      return cute::layout<1>(
        // divide complement to make it monotonic
        zipped_divide(make_layout(shape<i>(r2s_tv_tiler_shape)), complemented));
    });

    auto r2s_TV_D_tiler = left_inverse(take<0,2>(tE_D)).compose( // tv -> mn in epilogue tiler
      layout_gd_tv.compose( // tv -> mn in C
        make_layout(r2s_tv_tiler_shape).compose(r2s_tv_tiler))); // tv_tiler -> tv coord

    // G2R ----------------------------------------
    using TiledCopyG2R = TiledCopy<CopyAtomG2R, decltype(r2s_TV_D_tiler), EpilogueTileSize>;
    auto thread_g2r = TiledCopyG2R::get_slice(thread_idx);
    // auto tEgD = flat_divide(gC, EpilogueTile{});
    // auto tGR_gC = thread_g2r.partition_S(tEgD);
    auto cC = make_identity_tensor(make_shape(size<0>(gD), size<1>(gD))); // same shape as gC/gD
    auto cCt = flat_divide(cC, EpilogueTile{});                           
    auto tRS_cC = thread_g2r.partition_S(cCt);
    
    // R2S ----------------------------------------
    using TiledCopyR2S = TiledCopy<CopyAtomR2S, decltype(r2s_TV_D_tiler), EpilogueTileSize>;
    auto tiled_r2s = TiledCopyR2S{};

    auto tile_RS_R = 
      cute::layout<1>( // just V
        make_layout(
          make_shape(typename TiledCopyR2S::TiledNumThr{},typename TiledCopyR2S::TiledNumVal{}),make_stride(_0{},_1{}))
            .compose(group<1,3>(                            // V_copy -> V_fragment
              right_inverse(layout_gd_tv)                   // TV_copy -> TV_fragment
                .compose(tiled_r2s.tidfrg_S(tE_D)))))
                  (_,repeat<rank_v<decltype(tE_D)>>(_));    // TV_copy -> MN

    auto thread_r2s = tiled_r2s.get_slice(thread_idx);
    auto tErAcc = make_tensor(accumulators.data(), tile_RS_R);
    auto tRS_sAcc = thread_r2s.partition_D(sAcc);

    // S2R ----------------------------------------
    auto tiled_s2r = TiledCopyS2R{};
    auto thread_s2r = tiled_s2r.get_slice(thread_idx);
    auto tSR_sAcc = thread_s2r.partition_S(sAcc);
    auto tSR_rAcc = make_tensor<ElementD>(make_layout(tSR_sAcc.shape()));

    // R2R --------------------------------------------
    auto tiled_r2r = 
      make_tiled_copy(CopyAtomR2R{},typename CopyAtomR2R::ThrID{},make_layout(typename TiledCopyR2S::TiledNumVal{}));
    auto thread_r2r = tiled_r2r.get_slice(thread_idx);
    auto tRR_rAcc = make_tensor<ElementD>(tSR_rAcc.shape());
    auto tRR_rSrc = thread_r2r.retile_S(tSR_rAcc);
    auto tRR_rDst = thread_r2r.retile_D(tRR_rAcc);

    // R2G -------------------------------------------
    auto r2r_vlayout = filter(cute::layout<1>(tiled_r2r.get_layoutS_TV()).compose(cute::layout<1>(tiled_r2r.get_layoutD_TV())));
    auto r2g_tv_layout = (typename TiledCopyS2R::TiledLayout_TV{}).compose(_,r2r_vlayout);
    auto tiled_r2g = TiledCopy<CopyAtomR2G, decltype(r2g_tv_layout), EpilogueTileSize>{};
    auto thread_r2g = tiled_r2g.get_slice(thread_idx);

    auto tRG_rAcc = thread_r2g.retile_S(tRR_rDst);
    auto tEgD = flat_divide(gD, EpilogueTile{});
    auto tRG_gD = thread_r2g.partition_D(tEgD);

    // Repeat the D-partitioning for coordinates and predication
    Tensor cD   = make_identity_tensor(make_shape(size<0>(gD),size<1>(gD))); // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor cDt  = flat_divide(cD, EpilogueTile{});                           // (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor tRG_cD = thread_r2g.partition_D(cDt);

#if 0
    if (thread_idx == 0 && m_coord == 0 && n_coord == 0) {
      print("aC   : "); print(accumulators.layout()); print("\n");
      // print("gC   : "); print(gC.layout()); print("\n");
      print("gD   : "); print(gD.layout()); print("\n");
      // print("gBias   : "); print(gBias.layout()); print("\n");
      print("sAcc   : "); print(sAcc.layout()); print("\n");
      // print("rAcc   : "); print(rAcc.layout()); print("\n");
      print("\n");
      // print("tRS_rAcc : "); print(tRS_rAcc.layout()); print("\n");
      print("tRS_sAcc : "); print(tRS_sAcc.layout()); print("\n");
      print("\n");
      print("tSR_sAcc : "); print(tSR_sAcc.layout()); print("\n");
      print("tSR_rAcc : "); print(tSR_rAcc.layout()); print("\n");
      print("\n");
      print("tRR_rSrc : "); print(tRR_rSrc.layout()); print("\n");
      print("tRR_rDst : "); print(tRR_rDst.layout()); print("\n");
      print("\n");
      print("tRG_rAcc : "); print(tRG_rAcc.layout()); print("\n");
      print("tRG_gD   : "); print(tRG_gD.layout()); print("\n");
      print("tRS_cC   : "); print(tRS_cC.layout()); print("\n");
      print("cCt      : "); print(cCt.layout()); print("\n");
      print("cD       : "); print(cD.layout()); print("\n");
      print("tE_D     : "); print(tE_D.layout()); print("\n");
      print("sAcc     : "); print(sAcc.layout()); print("\n");
      print("tErAcc    : "); print(tErAcc.layout()); print("\n");
    }
#endif
    CUTLASS_PRAGMA_UNROLL
    for (int epi_tile_m = 0; epi_tile_m < size<2>(tEgD).value; ++epi_tile_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_tile_n = 0; epi_tile_n < size<3>(tEgD).value; ++epi_tile_n) {
        Tensor tRS_rAccmn = tErAcc(_,_,_,epi_tile_m,epi_tile_n); // ((2, (2, 2)), 1, 1)
        Tensor cC_mn = tRS_cC(_,_,_,epi_tile_m, epi_tile_n);
        Tensor tRS_rD = make_tensor_like<ElementD>(tRS_rAccmn);
                        
        int m_cta_coord_base = m_coord * size<0>(gD);
        int n_cta_coord_base = n_coord * size<1>(gD);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tRS_rAccmn); ++i) {
          auto cta_coord = cC_mn(i);
          tRS_rD(i) = epilogue_op(tRS_rAccmn(i),
                                  l_coord, 
                                  m_cta_coord_base + get<0>(cta_coord), 
                                  n_cta_coord_base + get<1>(cta_coord),
                                  elem_less(cta_coord, take<0,2>(residue_mnk)));
        }

        copy(CopyAtomR2S{}, tRS_rD, tRS_sAcc);
        copy(TiledCopyS2R{}, tSR_sAcc, tSR_rAcc);
        copy(CopyAtomR2R{}, tRR_rSrc, tRR_rDst);

        Tensor tRG_gDmn = tRG_gD(_,_,_,epi_tile_m,epi_tile_n);
        Tensor tRG_cDmn = tRG_cD(_,_,_,epi_tile_m,epi_tile_n);
        CUTLASS_PRAGMA_UNROLL
        for (int atom_i = 0; atom_i < size<0,1>(tRR_rAcc); ++atom_i) {
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tRR_rAcc); ++m) {
            CUTLASS_PRAGMA_UNROLL

            for (int n = 0; n < size<2>(tRR_rAcc); ++n) {
              
              if (elem_less(tRG_cDmn(make_coord(0,atom_i),m,n), take<0,2>(residue_mnk))) {
                copy(CopyAtomR2G{}, tRG_rAcc(make_coord(_,atom_i),m,n), tRG_gDmn(make_coord(_,atom_i),m,n));
              }
            }
          }
        }
      }
    }
  }

private:
  Params params;
  ThreadEpilogueOp epilogue_op;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
