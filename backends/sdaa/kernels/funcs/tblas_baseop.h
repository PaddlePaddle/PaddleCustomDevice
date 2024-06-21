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

#pragma once

#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "kernels/profiler/sdaa_wrapper.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

using Context = phi::CustomContext;

namespace tblas_ops {

__inline__ static bool isEnableHighPerformanceGemm() {
  int enable_high_performance_gemm = 0;
  const char* env = NULL;
  if ((env = std::getenv("HIGH_PERFORMANCE_GEMM")) != NULL) {
    enable_high_performance_gemm = atoi(env);
  }

  return enable_high_performance_gemm;
}

template <typename T>
struct TecoBlas;

template <>
struct TecoBlas<float> {
  template <typename... ARGS>
  static void Dot(const Context& dev_ctx,
                  int n,
                  const void* x,
                  int incx,
                  const void* y,
                  int incy,
                  void* result) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._n = n;
    param_pack._Atype = TECOBLAS_DATA_FLOAT;
    param_pack._Btype = TECOBLAS_DATA_FLOAT;
    param_pack._Ctype = TECOBLAS_DATA_FLOAT;
    param_pack._apiName = TECOBLAS_SDOT;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    std::stringstream ss;
    ss << "{"
       << "\"n\": " << n << ", \"incx\": " << incx << ", \"incy\": " << incy
       << "}";
    TBLAS_CHECK_WITH_MSG(tblasSdot(tblas_handle, n, x, incx, y, incy, result),
                         ss.str());
  }

  template <typename... ARGS>
  static void Gemv(const Context& dev_ctx,
                   tblasOperation_t trans,
                   int m,
                   int n,
                   float alpha,
                   float beta,
                   const void* A,
                   int lda,
                   const void* y,
                   int incy,
                   void* result,
                   int incr) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._transa = trans;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._alpha = alpha;
    param_pack._beta = beta;
    param_pack._lda = lda;
    param_pack._Atype = TECOBLAS_DATA_FLOAT;
    param_pack._Btype = TECOBLAS_DATA_FLOAT;
    param_pack._Ctype = TECOBLAS_DATA_FLOAT;
    param_pack._apiName = TECOBLAS_SGEMV;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    TBLAS_CHECK(tblasSgemv(
        tblas_handle, trans, m, n, alpha, A, lda, y, incy, beta, result, incr));
  }

  template <typename... ARGS>
  static void Gemm(const Context& dev_ctx,
                   tblasOperation_t trans_a,
                   tblasOperation_t trans_b,
                   int m,
                   int n,
                   int k,
                   float alpha,
                   const phi::dtype::float16* A,
                   int lda,
                   const phi::dtype::float16* B,
                   int ldb,
                   float beta,
                   float* C,
                   int ldc) {
    VLOG(4) << "use SDAA high performance GEMM";

    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._transa = trans_a;
    param_pack._transb = trans_b;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._alpha = alpha;
    param_pack._beta = beta;
    param_pack._lda = lda;
    param_pack._ldb = ldb;
    param_pack._ldc = ldc;
    param_pack._Atype = TECOBLAS_DATA_HALF;
    param_pack._Btype = TECOBLAS_DATA_HALF;
    param_pack._Ctype = TECOBLAS_DATA_FLOAT;
    param_pack._apiName = TECOBLAS_SGEMM_EX;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    TBLAS_CHECK_WITH_MSG(tecoblasSgemmEx(tblas_handle,
                                         trans_a,
                                         trans_b,
                                         m,
                                         n,
                                         k,
                                         alpha,
                                         A,
                                         param_pack._Atype,
                                         lda,
                                         B,
                                         param_pack._Btype,
                                         ldb,
                                         beta,
                                         C,
                                         param_pack._Ctype,
                                         ldc),
                         param_pack.get_formated_str());
  }

  template <typename... ARGS>
  static void Gemm(const Context& dev_ctx,
                   tblasOperation_t trans_a,
                   tblasOperation_t trans_b,
                   int m,
                   int n,
                   int k,
                   float alpha,
                   const float* A,
                   int lda,
                   const float* B,
                   int ldb,
                   float beta,
                   float* C,
                   int ldc) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._transa = trans_a;
    param_pack._transb = trans_b;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._alpha = alpha;
    param_pack._beta = beta;
    param_pack._lda = lda;
    param_pack._ldb = ldb;
    param_pack._ldc = ldc;
    param_pack._Atype = TECOBLAS_DATA_FLOAT;
    param_pack._Btype = TECOBLAS_DATA_FLOAT;
    param_pack._Ctype = TECOBLAS_DATA_FLOAT;
    param_pack._apiName = TECOBLAS_SGEMM;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    TBLAS_CHECK_WITH_MSG(tecoblasSgemm(tblas_handle,
                                       trans_a,
                                       trans_b,
                                       m,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       B,
                                       ldb,
                                       beta,
                                       C,
                                       ldc),
                         param_pack.get_formated_str());
  }

  template <typename... ARGS>
  static void GemmStridedBatched(const Context& dev_ctx,
                                 tblasOperation_t trans_a,
                                 tblasOperation_t trans_b,
                                 int m,
                                 int n,
                                 int k,
                                 float alpha,
                                 const phi::dtype::float16* A,
                                 int lda,
                                 int64_t stride_a,
                                 const phi::dtype::float16* B,
                                 int ldb,
                                 int64_t stride_b,
                                 float beta,
                                 float* C,
                                 int ldc,
                                 int64_t stride_c,
                                 int batch) {
    VLOG(4) << "use SDAA high performance GemmStridedBatched";

    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._transa = trans_a;
    param_pack._transb = trans_b;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._alpha = alpha;
    param_pack._beta = beta;
    param_pack._lda = lda;
    param_pack._ldb = ldb;
    param_pack._ldc = ldc;
    param_pack._Atype = TECOBLAS_DATA_HALF;
    param_pack._Btype = TECOBLAS_DATA_HALF;
    param_pack._Ctype = TECOBLAS_DATA_FLOAT;
    param_pack._strideA = stride_a;
    param_pack._strideB = stride_b;
    param_pack._strideC = stride_c;
    param_pack._batchCount = batch;
    param_pack._apiName = TECOBLAS_SGEMM_STRIDED_BATCHED_EX;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    TBLAS_CHECK_WITH_MSG(tecoblasSgemmStridedBatchedEx(tblas_handle,
                                                       trans_a,
                                                       trans_b,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       A,
                                                       param_pack._Atype,
                                                       lda,
                                                       stride_a,
                                                       B,
                                                       param_pack._Btype,
                                                       ldb,
                                                       stride_b,
                                                       beta,
                                                       C,
                                                       param_pack._Ctype,
                                                       ldc,
                                                       stride_c,
                                                       batch),
                         param_pack.get_formated_str());
  }

  template <typename... ARGS>
  static void GemmStridedBatched(const Context& dev_ctx,
                                 tblasOperation_t trans_a,
                                 tblasOperation_t trans_b,
                                 int m,
                                 int n,
                                 int k,
                                 float alpha,
                                 const float* A,
                                 int lda,
                                 int64_t stride_a,
                                 const float* B,
                                 int ldb,
                                 int64_t stride_b,
                                 float beta,
                                 float* C,
                                 int ldc,
                                 int64_t stride_c,
                                 int batch) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._transa = trans_a;
    param_pack._transb = trans_b;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._alpha = alpha;
    param_pack._beta = beta;
    param_pack._lda = lda;
    param_pack._ldb = ldb;
    param_pack._ldc = ldc;
    param_pack._Atype = TECOBLAS_DATA_FLOAT;
    param_pack._Btype = TECOBLAS_DATA_FLOAT;
    param_pack._Ctype = TECOBLAS_DATA_FLOAT;
    param_pack._strideA = stride_a;
    param_pack._strideB = stride_b;
    param_pack._strideC = stride_c;
    param_pack._apiName = TECOBLAS_SGEMM;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    std::string msg = param_pack.get_formated_str();
    for (size_t i = 0; i < batch; i++) {
      TBLAS_CHECK_WITH_MSG(tecoblasSgemm(tblas_handle,
                                         trans_a,
                                         trans_b,
                                         m,
                                         n,
                                         k,
                                         alpha,
                                         A + i * stride_a,
                                         lda,
                                         B + i * stride_b,
                                         ldb,
                                         beta,
                                         C + i * stride_c,
                                         ldc),
                           msg);
    }
  }

  template <typename... ARGS>
  static void GemmBatched(const Context& dev_ctx,
                          tblasOperation_t trans_a,
                          tblasOperation_t trans_b,
                          int m,
                          int n,
                          int k,
                          float alpha,
                          const phi::dtype::float16* A,
                          int lda,
                          const phi::dtype::float16* B,
                          int ldb,
                          float beta,
                          float* C,
                          int ldc,
                          int batch) {
    VLOG(4) << "use SDAA high performance GemmBatched";

    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._transa = trans_a;
    param_pack._transb = trans_b;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._alpha = alpha;
    param_pack._beta = beta;
    param_pack._lda = lda;
    param_pack._strideA = m * k;
    param_pack._ldb = ldb;
    param_pack._strideB = k * n;
    param_pack._ldc = ldc;
    param_pack._strideC = m * n;
    param_pack._Atype = TECOBLAS_DATA_HALF;
    param_pack._Btype = TECOBLAS_DATA_HALF;
    param_pack._Ctype = TECOBLAS_DATA_FLOAT;
    param_pack._batchCount = batch;
    param_pack._apiName = TECOBLAS_SGEMM_STRIDED_BATCHED_EX;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    TBLAS_CHECK_WITH_MSG(tecoblasSgemmStridedBatchedEx(tblas_handle,
                                                       trans_a,
                                                       trans_b,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       A,
                                                       param_pack._Atype,
                                                       lda,
                                                       param_pack._strideA,
                                                       B,
                                                       param_pack._Btype,
                                                       ldb,
                                                       param_pack._strideB,
                                                       beta,
                                                       C,
                                                       param_pack._Ctype,
                                                       ldc,
                                                       param_pack._strideC,
                                                       batch),
                         param_pack.get_formated_str());
  }

  template <typename... ARGS>
  static void GemmBatched(const Context& dev_ctx,
                          tblasOperation_t trans_a,
                          tblasOperation_t trans_b,
                          int m,
                          int n,
                          int k,
                          float alpha,
                          const float* A,
                          int lda,
                          const float* B,
                          int ldb,
                          float beta,
                          float* C,
                          int ldc,
                          int batch) {
    VLOG(4) << "use sgemm batched";
    GemmStridedBatched(dev_ctx,
                       trans_a,
                       trans_b,
                       m,
                       n,
                       k,
                       alpha,
                       A,
                       lda,
                       static_cast<int64_t>(m) * k,
                       B,
                       ldb,
                       static_cast<int64_t>(n) * k,
                       beta,
                       C,
                       ldc,
                       static_cast<int64_t>(m) * n,
                       batch);
  }

  template <typename... ARGS>
  static void Tril(const Context& dev_ctx,
                   int m,
                   int n,
                   int k,
                   int diagonal,
                   const void* x,
                   void* result) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._Atype = TECOBLAS_DATA_FLOAT;
    param_pack._Btype = TECOBLAS_DATA_FLOAT;
    param_pack._Ctype = TECOBLAS_DATA_FLOAT;
    param_pack._apiName = TECOBLAS_STRIL;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);

    void* x1 = const_cast<void*>(x);
    TBLAS_CHECK(tecoblasSTril(tblas_handle, m, n, k, diagonal, x1, result));
  }

  template <typename... ARGS>
  static void Triu(const Context& dev_ctx,
                   int m,
                   int n,
                   int k,
                   int diagonal,
                   const void* x,
                   void* result) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._Atype = TECOBLAS_DATA_FLOAT;
    param_pack._Btype = TECOBLAS_DATA_FLOAT;
    param_pack._Ctype = TECOBLAS_DATA_FLOAT;
    param_pack._apiName = TECOBLAS_STRIU;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    void* x1 = const_cast<void*>(x);
    TBLAS_CHECK(tecoblasSTriu(tblas_handle, m, n, k, diagonal, x1, result));
  }
};

template <>
struct TecoBlas<phi::dtype::float16> {
  template <typename... ARGS>
  static void Dot(const Context& dev_ctx,
                  int n,
                  const void* x,
                  int incx,
                  const void* y,
                  int incy,
                  void* result) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._n = n;
    param_pack._Atype = TECOBLAS_DATA_HALF;
    param_pack._Btype = TECOBLAS_DATA_HALF;
    param_pack._Ctype = TECOBLAS_DATA_HALF;
    param_pack._apiName = TECOBLAS_HDOT;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    TBLAS_CHECK(tblasHdot(tblas_handle, n, x, incx, y, incy, result));
  }

  template <typename... ARGS>
  static void Gemv(const Context& dev_ctx,
                   tblasOperation_t trans,
                   int m,
                   int n,
                   float alpha,
                   float beta,
                   const void* A,
                   int lda,
                   const void* y,
                   int incy,
                   void* result,
                   int incr) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._transa = trans;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._alpha = alpha;
    param_pack._beta = beta;
    param_pack._lda = lda;
    param_pack._Atype = TECOBLAS_DATA_HALF;
    param_pack._Btype = TECOBLAS_DATA_HALF;
    param_pack._Ctype = TECOBLAS_DATA_HALF;
    param_pack._apiName = TECOBLAS_HGEMV;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    TBLAS_CHECK(tblasHgemv(
        tblas_handle, trans, m, n, alpha, A, lda, y, incy, beta, result, incr));
  }

  template <typename... ARGS>
  static void Gemm(const Context& dev_ctx,
                   tblasOperation_t trans_a,
                   tblasOperation_t trans_b,
                   int m,
                   int n,
                   int k,
                   float alpha,
                   const void* A,
                   int lda,
                   const void* B,
                   int ldb,
                   float beta,
                   void* C,
                   int ldc) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._transa = trans_a;
    param_pack._transb = trans_b;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._alpha = alpha;
    param_pack._beta = beta;
    param_pack._lda = lda;
    param_pack._ldb = ldb;
    param_pack._ldc = ldc;
    param_pack._Atype = TECOBLAS_DATA_HALF;
    param_pack._Btype = TECOBLAS_DATA_HALF;
    param_pack._Ctype = TECOBLAS_DATA_HALF;
    param_pack._apiName = TECOBLAS_HGEMM;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    VLOG(4) << "use tblasHgemmV2";
    TBLAS_CHECK(tblasHgemmV2(tblas_handle,
                             trans_a,
                             trans_b,
                             m,
                             n,
                             k,
                             alpha,
                             A,
                             lda,
                             B,
                             ldb,
                             beta,
                             C,
                             ldc));
  }

  template <typename... ARGS>
  static void GemmStridedBatched(const Context& dev_ctx,
                                 tblasOperation_t trans_a,
                                 tblasOperation_t trans_b,
                                 int m,
                                 int n,
                                 int k,
                                 float alpha,
                                 const void* A,
                                 int lda,
                                 int64_t stride_a,
                                 const void* B,
                                 int ldb,
                                 int64_t stride_b,
                                 float beta,
                                 void* C,
                                 int ldc,
                                 int64_t stride_c,
                                 int batch) {
    VLOG(4) << "use hgemm strided batch";

    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._transa = trans_a;
    param_pack._transb = trans_b;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._alpha = alpha;
    param_pack._beta = beta;
    param_pack._lda = lda;
    param_pack._ldb = ldb;
    param_pack._ldc = ldc;
    param_pack._Atype = TECOBLAS_DATA_HALF;
    param_pack._Btype = TECOBLAS_DATA_HALF;
    param_pack._Ctype = TECOBLAS_DATA_HALF;
    param_pack._strideA = stride_a;
    param_pack._strideB = stride_b;
    param_pack._strideC = stride_c;
    param_pack._batchCount = batch;
    param_pack._apiName = TECOBLAS_HGEMM_STRIDED_BATCHED;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    TBLAS_CHECK(tblasHgemmStridedBatched(tblas_handle,
                                         trans_a,
                                         trans_b,
                                         m,
                                         n,
                                         k,
                                         alpha,
                                         A,
                                         lda,
                                         stride_a,
                                         B,
                                         ldb,
                                         stride_b,
                                         beta,
                                         C,
                                         ldc,
                                         stride_c,
                                         batch));
  }

  template <typename... ARGS>
  static void GemmBatched(const Context& dev_ctx,
                          tblasOperation_t trans_a,
                          tblasOperation_t trans_b,
                          int m,
                          int n,
                          int k,
                          float alpha,
                          const void* A,
                          int lda,
                          const void* B,
                          int ldb,
                          float beta,
                          void* C,
                          int ldc,
                          int batch) {
    VLOG(4) << "use hgemm batched V2";
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._transa = trans_a;
    param_pack._transb = trans_b;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._alpha = alpha;
    param_pack._beta = beta;
    param_pack._lda = lda;
    param_pack._strideA = m * k;
    param_pack._ldb = ldb;
    param_pack._strideB = n * k;
    param_pack._ldc = ldc;
    param_pack._strideC = m * n;
    param_pack._Atype = TECOBLAS_DATA_HALF;
    param_pack._Btype = TECOBLAS_DATA_HALF;
    param_pack._Ctype = TECOBLAS_DATA_HALF;
    param_pack._batchCount = batch;
    param_pack._apiName = TECOBLAS_HGEMM_STRIDED_BATCHED;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    TBLAS_CHECK(tecoblasHgemmStridedBatched(tblas_handle,
                                            trans_a,
                                            trans_b,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            A,
                                            lda,
                                            param_pack._strideA,
                                            B,
                                            ldb,
                                            param_pack._strideB,
                                            beta,
                                            C,
                                            ldc,
                                            param_pack._strideC,
                                            batch));
  }

  template <typename... ARGS>
  static void Tril(const Context& dev_ctx,
                   int m,
                   int n,
                   int k,
                   int diagonal,
                   const void* x,
                   void* result) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._Atype = TECOBLAS_DATA_HALF;
    param_pack._Btype = TECOBLAS_DATA_HALF;
    param_pack._Ctype = TECOBLAS_DATA_HALF;
    param_pack._apiName = TECOBLAS_HTRIL;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    void* x1 = const_cast<void*>(x);
    TBLAS_CHECK(tecoblasHTril(tblas_handle, m, n, k, diagonal, x1, result));
  }

  template <typename... ARGS>
  static void Triu(const Context& dev_ctx,
                   int m,
                   int n,
                   int k,
                   int diagonal,
                   const void* x,
                   void* result) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._Atype = TECOBLAS_DATA_HALF;
    param_pack._Btype = TECOBLAS_DATA_HALF;
    param_pack._Ctype = TECOBLAS_DATA_HALF;
    param_pack._apiName = TECOBLAS_HTRIU;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    void* x1 = const_cast<void*>(x);
    TBLAS_CHECK(tecoblasHTriu(tblas_handle, m, n, k, diagonal, x1, result));
  }
};

template <>
struct TecoBlas<double> {
  template <typename... ARGS>
  static void Tril(const Context& dev_ctx,
                   int m,
                   int n,
                   int k,
                   int diagonal,
                   const void* x,
                   void* result) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._Atype = TECOBLAS_DATA_DOUBLE;
    param_pack._Btype = TECOBLAS_DATA_DOUBLE;
    param_pack._Ctype = TECOBLAS_DATA_DOUBLE;
    param_pack._apiName = TECOBLAS_DTRIL;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    void* x1 = const_cast<void*>(x);
    TBLAS_CHECK(tecoblasDTril(tblas_handle, m, n, k, diagonal, x1, result));
  }

  template <typename... ARGS>
  static void Triu(const Context& dev_ctx,
                   int m,
                   int n,
                   int k,
                   int diagonal,
                   const void* x,
                   void* result) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._Atype = TECOBLAS_DATA_DOUBLE;
    param_pack._Btype = TECOBLAS_DATA_DOUBLE;
    param_pack._Ctype = TECOBLAS_DATA_DOUBLE;
    param_pack._apiName = TECOBLAS_DTRIU;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    void* x1 = const_cast<void*>(x);
    TBLAS_CHECK(tecoblasDTriu(tblas_handle, m, n, k, diagonal, x1, result));
  }
};

template <>
struct TecoBlas<int64_t> {
  template <typename... ARGS>
  static void Tril(const Context& dev_ctx,
                   int m,
                   int n,
                   int k,
                   int diagonal,
                   const void* x,
                   void* result) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._Atype = TECOBLAS_DATA_INT64;
    param_pack._Btype = TECOBLAS_DATA_INT64;
    param_pack._Ctype = TECOBLAS_DATA_INT64;
    param_pack._apiName = TECOBLAS_DTRIL;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    void* x1 = const_cast<void*>(x);
    TBLAS_CHECK(tecoblasDTril(tblas_handle, m, n, k, diagonal, x1, result));
  }

  template <typename... ARGS>
  static void Triu(const Context& dev_ctx,
                   int m,
                   int n,
                   int k,
                   int diagonal,
                   const void* x,
                   void* result) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    param_pack._Atype = TECOBLAS_DATA_INT64;
    param_pack._Btype = TECOBLAS_DATA_INT64;
    param_pack._Ctype = TECOBLAS_DATA_INT64;
    param_pack._apiName = TECOBLAS_DTRIU;
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    void* x1 = const_cast<void*>(x);
    TBLAS_CHECK(tecoblasDTriu(tblas_handle, m, n, k, diagonal, x1, result));
  }
};

template <>
struct TecoBlas<bool> {
  template <typename... ARGS>
  static void Tril(const Context& dev_ctx,
                   int m,
                   int n,
                   int k,
                   int diagonal,
                   const void* x,
                   void* result) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    /* need to check bool dtype name in tecoblas. */
    param_pack._Atype = TECOBLAS_DATA_BOOL;
    param_pack._Btype = TECOBLAS_DATA_BOOL;
    param_pack._Ctype = TECOBLAS_DATA_BOOL;
    param_pack._apiName = TECOBLAS_BTRIL;  // need to check API name.
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    void* x1 = const_cast<void*>(x);
    TBLAS_CHECK(tecoblasBTril(
        tblas_handle, m, n, k, diagonal, x1, result));  // need to check API
                                                        // name.
  }

  template <typename... ARGS>
  static void Triu(const Context& dev_ctx,
                   int m,
                   int n,
                   int k,
                   int diagonal,
                   const void* x,
                   void* result) {
    phi::DenseTensor workspace;
    struct MatmulParam param_pack;
    param_pack._m = m;
    param_pack._n = n;
    param_pack._k = k;
    /* need to check bool dtype name in tecoblas. */
    param_pack._Atype = TECOBLAS_DATA_BOOL;
    param_pack._Btype = TECOBLAS_DATA_BOOL;
    param_pack._Ctype = TECOBLAS_DATA_BOOL;
    param_pack._apiName = TECOBLAS_BTRIU;  // need to check API name.
    tblasHandle_t tblas_handle =
        GetBlasHandleFromCTX(dev_ctx, param_pack, &workspace);
    void* x1 = const_cast<void*>(x);
    TBLAS_CHECK(tecoblasBTriu(
        tblas_handle, m, n, k, diagonal, x1, result));  // need to check API
                                                        // name.
  }
};

inline void ReshpaeToMatrix(const int num_col_dims,
                            const std::vector<int64_t>& dims,
                            std::vector<int64_t>* matrix_dims) {
  int ndim = 2;
  (*matrix_dims).resize(ndim);
  int64_t first_dims = 1;
  int64_t second_dims = 1;
  for (int i = 0; i < num_col_dims; i++) {
    first_dims *= dims[i];
  }
  (*matrix_dims)[0] = first_dims;
  for (int j = 0; j < dims.size() - num_col_dims; j++) {
    second_dims *= dims[j + num_col_dims];
  }
  (*matrix_dims)[1] = second_dims;
}

template <typename T>
void doBroadcastTo(const T* x_ptr,
                   const T* y_ptr,
                   const T* out_ptr,
                   std::vector<int64_t>* x_dims,
                   std::vector<int64_t>* y_dims,
                   std::vector<int64_t>* out_dims,
                   std::vector<T*>* a,
                   std::vector<T*>* b,
                   std::vector<T*>* c) {
  T* x_ptr_copy = const_cast<T*>(x_ptr);
  T* y_ptr_copy = const_cast<T*>(y_ptr);
  T* out_ptr_copy = const_cast<T*>(out_ptr);
  assert((*y_dims).size() == (*out_dims).size() &&
         (*x_dims).size() == (*out_dims).size());
  int n_dims = (*out_dims).size() - 2;
  std::vector<int64_t> a_stride, b_stride, c_stride;
  a_stride.resize(n_dims);
  b_stride.resize(n_dims);
  c_stride.resize(n_dims);
  a_stride[n_dims - 1] = 1;
  b_stride[n_dims - 1] = 1;
  c_stride[n_dims - 1] = 1;
  for (int i = n_dims - 2; i >= 0; --i) {
    a_stride[i] = a_stride[i + 1] * (*x_dims)[i + 1];
    b_stride[i] = b_stride[i + 1] * (*y_dims)[i + 1];
    c_stride[i] = c_stride[i + 1] * (*out_dims)[i + 1];
  }
  int B = 1;
  for (int i = 0; i < (*out_dims).size() - 2; ++i) {
    B *= (*out_dims)[i];
  }
  (*a).resize(B), (*b).resize(B), (*c).resize(B);
  int a_mat_size =
      (*x_dims)[(*x_dims).size() - 1] * (*x_dims)[(*x_dims).size() - 2];
  int b_mat_size =
      (*y_dims)[(*y_dims).size() - 1] * (*y_dims)[(*y_dims).size() - 2];
  int c_mat_size =
      (*out_dims)[(*out_dims).size() - 1] * (*out_dims)[(*out_dims).size() - 2];
  for (int i = 0; i < B; ++i) {
    int tmp_i = i;
    int a_i = 0, b_i = 0;
    for (int j = (*out_dims).size() - 3; j >= 0; --j) {
      int out_i = tmp_i % (*out_dims)[j];
      tmp_i = tmp_i / (*out_dims)[j];
      a_i += ((*x_dims)[j] == 1 ? 0 : out_i) * a_stride[j];
      b_i += ((*y_dims)[j] == 1 ? 0 : out_i) * b_stride[j];
    }
    (*a)[i] = x_ptr_copy + a_mat_size * a_i;
    (*b)[i] = y_ptr_copy + b_mat_size * b_i;
    (*c)[i] = out_ptr_copy + c_mat_size * i;
  }
}

template <typename T>
void Dot(const Context& dev_ctx,
         const phi::DenseTensor& X,
         const phi::DenseTensor& Y,
         phi::DenseTensor* out) {
  sdaa_ops::doMemsetTensor(dev_ctx, static_cast<int>(0), out);

  int n = X.numel();
  int incx = 1, incy = 1;

  if (out->dtype() == phi::DataType::FLOAT16) {
    phi::DenseTensor out_float;
    out_float.Resize(out->dims());
    dev_ctx.template Alloc<float>(&out_float);

    TecoBlas<T>::Dot(
        dev_ctx, n, X.data(), incx, Y.data(), incy, out_float.data());

    sdaa_ops::doCastTensor(dev_ctx, out_float, out);
  } else {
    TecoBlas<T>::Dot(dev_ctx, n, X.data(), incx, Y.data(), incy, out->data());
  }
}

template <typename T>
void MatVec(const Context& dev_ctx,
            const phi::DenseTensor& X,
            const phi::DenseTensor& Y,
            const bool transpose_x,
            phi::DenseTensor* out) {
  sdaa_ops::doMemsetTensor(dev_ctx, static_cast<int>(0), out);

  std::vector<int64_t> X_Dims = phi::vectorize<int64_t>(X.dims());

  int m = X_Dims[0];
  int n = X_Dims[1];
  float alpha = 1.0;
  int incy = 1;
  float beta = 0.0;
  int incr = 1;
  tblasOperation_t trans_x = TBLAS_OP_N;
  int lda = n;
  if (transpose_x) {
    trans_x = TBLAS_OP_T;
  }

  TecoBlas<T>::Gemv(dev_ctx,
                    trans_x,
                    m,
                    n,
                    alpha,
                    beta,
                    X.data(),
                    lda,
                    Y.data(),
                    incy,
                    out->data(),
                    incr);
}

template <typename T>
void MatMul2D(const Context& dev_ctx,
              const phi::DenseTensor& X,
              const phi::DenseTensor& Y,
              const bool transpose_x,
              const bool transpose_y,
              phi::DenseTensor* out) {
  if (out->dtype() == phi::DataType::FLOAT32) {
    sdaa_ops::doMemsetTensor(dev_ctx, static_cast<int>(0), out);
  }

  std::vector<int64_t> X_Dims = phi::vectorize<int64_t>(X.dims());
  std::vector<int64_t> Y_Dims = phi::vectorize<int64_t>(Y.dims());

  int x_row = X_Dims[0];
  int x_col = X_Dims[1];
  int y_row = Y_Dims[0];
  int y_col = Y_Dims[1];
  int m = x_row;
  int k = x_col;
  int n = y_col;
  float alpha = 1.0;
  float beta = 0.0;
  int lda = k;
  int ldb = n;
  int ldc = n;

  tblasOperation_t trans_x = TBLAS_OP_N;
  tblasOperation_t trans_y = TBLAS_OP_N;
  if (transpose_x) {
    trans_x = TBLAS_OP_T;
    m = x_col;
    k = x_row;
  }
  if (transpose_y) {
    trans_y = TBLAS_OP_T;
    n = y_row;
    ldc = n;
  }

  if (out->dtype() == phi::DataType::FLOAT32 && isEnableHighPerformanceGemm()) {
    phi::DenseTensor temp_x;
    phi::DenseTensorMeta temp_x_meta = {phi::DataType::FLOAT16, X.dims()};
    temp_x.set_meta(temp_x_meta);
    dev_ctx.Alloc<phi::dtype::float16>(&temp_x);
    sdaa_ops::doCastTensor(dev_ctx, X, &temp_x);

    phi::DenseTensor temp_y;
    phi::DenseTensorMeta temp_y_meta = {phi::DataType::FLOAT16, Y.dims()};
    temp_y.set_meta(temp_y_meta);
    dev_ctx.Alloc<phi::dtype::float16>(&temp_y);
    sdaa_ops::doCastTensor(dev_ctx, Y, &temp_y);

    TecoBlas<float>::Gemm(dev_ctx,
                          trans_x,
                          trans_y,
                          m,
                          n,
                          k,
                          alpha,
                          temp_x.data<phi::dtype::float16>(),
                          lda,
                          temp_y.data<phi::dtype::float16>(),
                          ldb,
                          beta,
                          out->data<float>(),
                          ldc);
  } else {
    TecoBlas<T>::Gemm(dev_ctx,
                      trans_x,
                      trans_y,
                      m,
                      n,
                      k,
                      alpha,
                      X.data<T>(),
                      lda,
                      Y.data<T>(),
                      ldb,
                      beta,
                      out->data<T>(),
                      ldc);
  }
}

template <typename T>
void MatMulND(const Context& dev_ctx,
              const std::vector<T*>& X,
              const std::vector<int64_t>& x_mat_dims,
              const std::vector<T*>& Y,
              const std::vector<int64_t>& y_mat_dims,
              const std::vector<T*>& result,
              const bool transpose_x,
              const bool transpose_y,
              phi::DenseTensor* out) {
  if (out->dtype() == phi::DataType::FLOAT32) {
    sdaa_ops::doMemsetTensor(dev_ctx, static_cast<int>(0), out);
  }

  assert(X.size() == result.size() && Y.size() == result.size());
  int x_mat_row = x_mat_dims[0];
  int x_mat_col = x_mat_dims[1];
  int y_mat_row = y_mat_dims[0];
  int y_mat_col = y_mat_dims[1];
  int m = x_mat_row;
  int k = x_mat_col;
  int n = y_mat_col;
  float alpha = 1.0;
  float beta = 0.0;
  int lda = k;
  int ldb = n;
  int ldc = n;

  tblasOperation_t trans_x = TBLAS_OP_N;
  tblasOperation_t trans_y = TBLAS_OP_N;
  if (transpose_x) {
    trans_x = TBLAS_OP_T;
    m = x_mat_col;
    k = x_mat_row;
  }
  if (transpose_y) {
    trans_y = TBLAS_OP_T;
    n = y_mat_row;
    ldc = n;
  }
  for (int i = 0; i < X.size(); i++) {
    TecoBlas<T>::Gemm(dev_ctx,
                      trans_x,
                      trans_y,
                      m,
                      n,
                      k,
                      alpha,
                      X[i],
                      lda,
                      Y[i],
                      ldb,
                      beta,
                      result[i],
                      ldc);
  }
}

template <typename T>
void DotGradFunction(const Context& dev_ctx,
                     const phi::DenseTensor& dout,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* dy) {
  sdaa_ops::doMemsetTensor(dev_ctx, static_cast<int>(0), dy);
  sdaa_ops::doElementMul(dev_ctx, dout, x, -1, dy);
}

template <typename T>
void SingleMatmulWithBatchedMat(const Context& dev_ctx,
                                const phi::DenseTensor& X,
                                const phi::DenseTensor& Y,
                                const bool transpose_x,
                                const bool transpose_y,
                                phi::DenseTensor* out) {
  VLOG(4) << "mat x batched mat";
  auto x_dims = X.dims();
  auto y_dims = Y.dims();
  int x_dims_size = x_dims.size();
  int y_dims_size = y_dims.size();
  // Now this func focus on [weight @ input] situation, so x_dim must be 2
  PADDLE_ENFORCE_EQ(x_dims_size == 2 && y_dims_size > 2,
                    true,
                    phi::errors::InvalidArgument(
                        "X'dims size must be equal to %d and Y'dims size "
                        "must be greater than %d in SingleMatmulWithBatchedMat"
                        "But received X'dims size is %d and Y'dims size is %d",
                        2,
                        2,
                        x_dims_size,
                        y_dims_size));
  auto m = x_dims[transpose_x ? 1 : 0];
  auto k = x_dims[transpose_x ? 0 : 1];
  auto n = y_dims[transpose_y ? y_dims_size - 2 : y_dims_size - 1];
  int lda = transpose_x ? m : k;
  int ldb = transpose_y ? k : n;
  int ldc = n;
  int stride_a = 0;
  int stride_b = k * n;
  int stride_c = m * n;
  tblasOperation_t trans_x = transpose_x ? TBLAS_OP_T : TBLAS_OP_N;
  tblasOperation_t trans_y = transpose_y ? TBLAS_OP_T : TBLAS_OP_N;
  float alpha = 1.0;
  float beta = 0.0;
  int batch = phi::product(phi::slice_ddim(y_dims, 0, y_dims_size - 2));

  if (out->dtype() == phi::DataType::FLOAT32 && isEnableHighPerformanceGemm()) {
    phi::DenseTensor temp_x;
    phi::DenseTensorMeta temp_x_meta = {phi::DataType::FLOAT16, X.dims()};
    temp_x.set_meta(temp_x_meta);
    dev_ctx.Alloc<phi::dtype::float16>(&temp_x);
    sdaa_ops::doCastTensor(dev_ctx, X, &temp_x);

    phi::DenseTensor temp_y;
    phi::DenseTensorMeta temp_y_meta = {phi::DataType::FLOAT16, Y.dims()};
    temp_y.set_meta(temp_y_meta);
    dev_ctx.Alloc<phi::dtype::float16>(&temp_y);
    sdaa_ops::doCastTensor(dev_ctx, Y, &temp_y);

    TecoBlas<float>::GemmStridedBatched(dev_ctx,
                                        trans_x,
                                        trans_y,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        temp_x.data<phi::dtype::float16>(),
                                        lda,
                                        stride_a,
                                        temp_y.data<phi::dtype::float16>(),
                                        ldb,
                                        stride_b,
                                        beta,
                                        out->data<float>(),
                                        ldc,
                                        stride_c,
                                        batch);
  } else {
    TecoBlas<T>::GemmStridedBatched(dev_ctx,
                                    trans_x,
                                    trans_y,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    X.data<T>(),
                                    lda,
                                    stride_a,
                                    Y.data<T>(),
                                    ldb,
                                    stride_b,
                                    beta,
                                    out->data<T>(),
                                    ldc,
                                    stride_c,
                                    batch);
  }
}

template <typename T>
void BatchedMatmulWithSingleMat(const Context& dev_ctx,
                                const phi::DenseTensor& X,
                                const phi::DenseTensor& Y,
                                const bool transpose_x,
                                const bool transpose_y,
                                phi::DenseTensor* out) {
  VLOG(4) << "batched mat x mat";
  auto x_dims = X.dims();
  auto y_dims = Y.dims();
  int x_dims_size = x_dims.size();
  int y_dims_size = y_dims.size();
  // Now this func focus on [input @ weight] situation, so y_dim must be 2
  PADDLE_ENFORCE_EQ(x_dims_size > 2 && y_dims_size == 2,
                    true,
                    phi::errors::InvalidArgument(
                        "X'dims size must be greater than %d and Y'dims size "
                        "must be equal to %d in BatchedMatmulWithSingleMat"
                        "But received X'dims size is %d and Y'dims size is %d",
                        2,
                        2,
                        x_dims_size,
                        y_dims_size));

  // when trans_x is not needed, we fuse its all non-matrix dims and do normal
  // matmul
  if (!transpose_x) {
    VLOG(4) << "Matrix A-dimension fusion";
    int fused_dims = phi::product(phi::slice_ddim(x_dims, 0, x_dims_size - 1));
    phi::DenseTensor x_temp(X);
    x_temp.Resize({fused_dims, x_dims[x_dims_size - 1]});
    tblas_ops::MatMul2D<T>(dev_ctx, x_temp, Y, transpose_x, transpose_y, out);
    return;
  }
  auto m = x_dims[x_dims_size - 1];
  auto k = y_dims[transpose_y ? 1 : 0];
  auto n = y_dims[transpose_y ? 0 : 1];
  int lda = m;
  int ldb = transpose_y ? k : n;
  int ldc = n;
  int stride_a = m * k;
  int stride_b = 0;
  int stride_c = m * n;
  tblasOperation_t trans_x = TBLAS_OP_T;
  tblasOperation_t trans_y = transpose_y ? TBLAS_OP_T : TBLAS_OP_N;
  float alpha = 1.0;
  float beta = 0.0;
  int batch = phi::product(phi::slice_ddim(x_dims, 0, x_dims_size - 2));

  if (out->dtype() == phi::DataType::FLOAT32 && isEnableHighPerformanceGemm()) {
    phi::DenseTensor temp_x;
    phi::DenseTensorMeta temp_x_meta = {phi::DataType::FLOAT16, X.dims()};
    temp_x.set_meta(temp_x_meta);
    dev_ctx.Alloc<phi::dtype::float16>(&temp_x);
    sdaa_ops::doCastTensor(dev_ctx, X, &temp_x);

    phi::DenseTensor temp_y;
    phi::DenseTensorMeta temp_y_meta = {phi::DataType::FLOAT16, Y.dims()};
    temp_y.set_meta(temp_y_meta);
    dev_ctx.Alloc<phi::dtype::float16>(&temp_y);
    sdaa_ops::doCastTensor(dev_ctx, Y, &temp_y);

    TecoBlas<float>::GemmStridedBatched(dev_ctx,
                                        trans_x,
                                        trans_y,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        temp_x.data<phi::dtype::float16>(),
                                        lda,
                                        stride_a,
                                        temp_y.data<phi::dtype::float16>(),
                                        ldb,
                                        stride_b,
                                        beta,
                                        out->data<float>(),
                                        ldc,
                                        stride_c,
                                        batch);
  } else {
    TecoBlas<T>::GemmStridedBatched(dev_ctx,
                                    trans_x,
                                    trans_y,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    X.data<T>(),
                                    lda,
                                    stride_a,
                                    Y.data<T>(),
                                    ldb,
                                    stride_b,
                                    beta,
                                    out->data<T>(),
                                    ldc,
                                    stride_c,
                                    batch);
  }
}

template <typename T>
void BatchMatmul(const Context& dev_ctx,
                 const phi::DenseTensor& X,
                 const phi::DenseTensor& Y,
                 const bool transpose_x,
                 const bool transpose_y,
                 phi::DenseTensor* out) {
  std::vector<int64_t> X_Dims = phi::vectorize<int64_t>(X.dims());
  std::vector<int64_t> Y_Dims = phi::vectorize<int64_t>(Y.dims());
  int X_Dims_size = X_Dims.size();
  int Y_Dims_size = Y_Dims.size();

  int m = X_Dims[transpose_x ? X_Dims_size - 1 : X_Dims_size - 2];
  int k = X_Dims[transpose_x ? X_Dims_size - 2 : X_Dims_size - 1];
  int n = Y_Dims[transpose_y ? Y_Dims_size - 2 : Y_Dims_size - 1];
  int lda = transpose_x ? m : k;
  int ldb = transpose_y ? k : n;
  int ldc = n;
  int batchcount = std::accumulate(
      X_Dims.begin(), X_Dims.end() - 2, 1, std::multiplies<int>());
  tblasOperation_t trans_x = transpose_x ? TBLAS_OP_T : TBLAS_OP_N;
  tblasOperation_t trans_y = transpose_y ? TBLAS_OP_T : TBLAS_OP_N;

  float alpha = 1.0;
  float beta = 0.0;

  if (out->dtype() == phi::DataType::FLOAT32 && isEnableHighPerformanceGemm()) {
    phi::DenseTensor temp_x;
    phi::DenseTensorMeta temp_x_meta = {phi::DataType::FLOAT16, X.dims()};
    temp_x.set_meta(temp_x_meta);
    dev_ctx.Alloc<phi::dtype::float16>(&temp_x);
    sdaa_ops::doCastTensor(dev_ctx, X, &temp_x);

    phi::DenseTensor temp_y;
    phi::DenseTensorMeta temp_y_meta = {phi::DataType::FLOAT16, Y.dims()};
    temp_y.set_meta(temp_y_meta);
    dev_ctx.Alloc<phi::dtype::float16>(&temp_y);
    sdaa_ops::doCastTensor(dev_ctx, Y, &temp_y);

    TecoBlas<float>::GemmBatched(dev_ctx,
                                 trans_x,
                                 trans_y,
                                 m,
                                 n,
                                 k,
                                 alpha,
                                 temp_x.data<phi::dtype::float16>(),
                                 lda,
                                 temp_y.data<phi::dtype::float16>(),
                                 ldb,
                                 beta,
                                 out->data<float>(),
                                 ldc,
                                 batchcount);
  } else {
    TecoBlas<T>::GemmBatched(dev_ctx,
                             trans_x,
                             trans_y,
                             m,
                             n,
                             k,
                             alpha,
                             X.data<T>(),
                             lda,
                             Y.data<T>(),
                             ldb,
                             beta,
                             out->data<T>(),
                             ldc,
                             batchcount);
  }
}

}  // namespace tblas_ops
}  // namespace custom_kernel
