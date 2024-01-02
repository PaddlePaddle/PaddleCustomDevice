// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/phi_funcs.h"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T>
void GEMM(bool trans_x,
          bool trans_y,
          size_t M,
          size_t K,
          size_t N,
          const T* x,
          const T* y,
          T* out,
          bool trans_out = false) {
  memset(out, 0, M * N * sizeof(T));
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      auto* out_data = trans_out ? &out[n * M + m] : &out[m * N + n];
      for (size_t k = 0; k < K; ++k) {
        auto x_dat = trans_x ? x[k * M + m] : x[m * K + k];
        auto y_dat = trans_y ? y[n * K + k] : y[k * N + n];
        *out_data += x_dat * y_dat;
      }
    }
  }
}

template <typename T>
void BatchedGEMM(bool trans_x,
                 bool trans_y,
                 size_t M,
                 size_t K,
                 size_t N,
                 const T* x,
                 const T* y,
                 T* out,
                 size_t batch_size,
                 bool x_is_larger,
                 bool trans_out = false,
                 bool bs_flag = false,
                 bool reduce_bs = false,
                 float alpha = 1.0) {
  memset(out, 0, sizeof(T) * (reduce_bs ? M * N : batch_size * M * N));
  if (x_is_larger) {
    for (size_t bs = 0; bs < batch_size; ++bs) {
      for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
          auto out_bs = reduce_bs ? 0 : bs * M * N;
          auto* out_data =
              trans_out ? &out[out_bs + n * M + m] : &out[out_bs + m * N + n];
          for (size_t k = 0; k < K; ++k) {
            auto x_dat =
                trans_x ? x[bs * M * K + k * M + m] : x[bs * M * K + m * K + k];
            auto y_bs = bs_flag ? bs * N * K : 0;
            auto y_dat = trans_y ? y[y_bs + n * K + k] : y[y_bs + k * N + n];
            *out_data += static_cast<T>(alpha) * (x_dat * y_dat);
          }
        }
      }
    }
  } else {
    for (size_t bs = 0; bs < batch_size; ++bs) {
      for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
          auto out_bs = reduce_bs ? 0 : bs * M * N;
          auto* out_data =
              trans_out ? &out[out_bs + n * M + m] : &out[out_bs + m * N + n];
          for (size_t k = 0; k < K; ++k) {
            auto y_dat =
                trans_y ? y[bs * K * N + n * K + k] : y[bs * K * N + k * N + n];
            auto x_bs = bs_flag ? bs * M * K : 0;
            auto x_dat = trans_x ? x[x_bs + k * M + m] : x[x_bs + m * K + k];
            *out_data += static_cast<T>(alpha) * (x_dat * y_dat);
          }
        }
      }
    }
  }
}

template <typename T>
void MatmulKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto x_data = x.data<T>();
  auto y_data = y.data<T>();
  auto x_ndim = x_dims.size();
  auto y_ndim = y_dims.size();

  if (x_ndim == 1 && y_ndim == 1) {
    auto M = x.numel();
    auto N = y.numel();
    PD_CHECK(M == N, "M must be equal to N.");
    out->Resize({1});
    auto out_data = dev_ctx.template Alloc<T>(out);
    GEMM(false, false, 1, M, 1, x.data<T>(), y.data<T>(), out_data);
    return;
  } else if (x_ndim == 1) {
    auto M = 1;
    auto K = x.numel();
    auto N = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
    if (transpose_y) {
      PD_CHECK(K == y_dims[y_ndim - 1],
               "Input(Y) has error dim."
               "Y'dims[%d] must be equal to %d"
               "But received Y'dims[%d] is %d",
               y_ndim - 1,
               K,
               y_ndim - 1,
               y_dims[y_ndim - 1]);
    } else {
      PD_CHECK(K == y_dims[y_ndim - 2],
               "Input(Y) has error dim."
               "Y'dims[%d] must be equal to %d"
               "But received Y'dims[%d] is %d",
               y_ndim - 2,
               K,
               y_ndim - 2,
               y_dims[y_ndim - 2]);
    }
    std::vector<int64_t> out_dims(y_ndim - 1);
    if (transpose_y) {
      std::copy_n(y_dims.cbegin(), y_ndim - 1, out_dims.begin());
    } else {
      std::copy_n(y_dims.cbegin(), y_ndim - 2, out_dims.begin());
      out_dims.back() = y_dims.back();
    }
    out->Resize(out_dims);
    auto out_data = dev_ctx.template Alloc<T>(out);
    if (y_ndim == 2) {
      GEMM(false, transpose_y, M, K, N, x_data, y_data, out_data);
    } else {
      BatchedGEMM(false,
                  transpose_y,
                  M,
                  K,
                  N,
                  x_data,
                  y_data,
                  out_data,
                  y_dims[0],
                  false);
    }
    return;
  } else if (y_ndim == 1) {
    auto M = transpose_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
    auto K = y.numel();
    auto N = 1;
    if (!transpose_x) {
      PD_CHECK(K == x_dims[x_ndim - 1],
               "Input(Y) has error dim."
               "Y'dims[%d] must be equal to %d"
               "But received Y'dims[%d] is %d",
               x_ndim - 1,
               K,
               x_ndim - 1,
               x_dims[x_ndim - 1]);
    } else {
      PD_CHECK(K == x_dims[x_ndim - 2],
               "Input(Y) has error dim."
               "Y'dims[%d] must be equal to %d"
               "But received Y'dims[%d] is %d",
               x_ndim - 2,
               K,
               x_ndim - 2,
               x_dims[x_ndim - 2]);
    }
    std::vector<int64_t> out_dims(x_ndim - 1);
    if (!transpose_x) {
      std::copy_n(x_dims.cbegin(), x_ndim - 1, out_dims.begin());
    } else {
      std::copy_n(x_dims.cbegin(), x_ndim - 2, out_dims.begin());
      out_dims.back() = x_dims.back();
    }
    out->Resize(out_dims);
    auto out_data = dev_ctx.template Alloc<T>(out);
    if (x_ndim == 2) {
      GEMM(transpose_x, false, M, K, N, x_data, y_data, out_data);
    } else {
      BatchedGEMM(transpose_x,
                  false,
                  M,
                  K,
                  N,
                  x_data,
                  y_data,
                  out_data,
                  x_dims[0],
                  true);
    }
    return;
  }

  if (x_ndim == 2 && y_ndim == 2) {
    auto M = transpose_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
    auto K = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
    auto N = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
    std::vector<int64_t> out_dims({M, N});
    out->Resize(out_dims);
    auto out_data = dev_ctx.template Alloc<T>(out);
    GEMM(transpose_x, transpose_y, M, K, N, x_data, y_data, out_data);
  } else if (x_ndim == 3) {
    auto M = transpose_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
    auto K = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
    auto N = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
    if (transpose_y) {
      PD_CHECK(y_dims[y_ndim - 1] == K,
               "Input(Y) has error dim."
               "Y'dims[%d] must be equal to %d"
               "But received Y'dims[%d] is %d",
               y_ndim - 1,
               K,
               y_ndim - 1,
               y_dims[y_ndim - 1]);
    } else {
      PD_CHECK(y_dims[y_ndim - 2] == K,
               "Input(Y) has error dim."
               "Y'dims[%d] must be equal to %d"
               "But received Y'dims[%d] is %d",
               y_ndim - 2,
               K,
               y_ndim - 2,
               y_dims[y_ndim - 2]);
    }
    std::vector<int64_t> out_dims({x_dims[0], M, N});
    out->Resize(out_dims);
    auto out_data = dev_ctx.template Alloc<T>(out);
    BatchedGEMM(transpose_x,
                transpose_y,
                M,
                K,
                N,
                x_data,
                y_data,
                out_data,
                x_dims[0],
                true);
  } else if (y_ndim == 3) {
    auto M = transpose_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
    auto K = transpose_y ? y_dims[y_ndim - 1] : y_dims[y_ndim - 2];
    auto N = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
    if (!transpose_x) {
      PD_CHECK(x_dims[x_ndim - 1] == K,
               "Input(X) has error dim."
               "X'dims[%d] must be equal to %d"
               "But received X'dims[%d] is %d",
               x_ndim - 1,
               K,
               x_ndim - 1,
               x_dims[x_ndim - 1]);
    } else {
      PD_CHECK(x_dims[x_ndim - 2] == K,
               "Input(X) has error dim."
               "X'dims[%d] must be equal to %d"
               "But received X'dims[%d] is %d",
               x_ndim - 2,
               K,
               x_ndim - 2,
               x_dims[y_ndim - 2]);
    }
    std::vector<int64_t> out_dims({y_dims[0], M, N});
    out->Resize(out_dims);
    auto out_data = dev_ctx.template Alloc<T>(out);
    BatchedGEMM(transpose_x,
                transpose_y,
                M,
                K,
                N,
                x_data,
                y_data,
                out_data,
                y_dims[0],
                false);
  }
}

template <typename T>
void MatmulGradKernel(const phi::Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      const phi::DenseTensor& out_grad,
                      bool transpose_x,
                      bool transpose_y,
                      phi::DenseTensor* dx,
                      phi::DenseTensor* dy) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto dout_dims = out_grad.dims();

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int ndim = dout_dims.size();

  auto x_data = x.data<T>();
  auto y_data = y.data<T>();
  auto out_grad_data = out_grad.data<T>();
  auto dx_data = static_cast<T*>(nullptr);
  auto dy_data = static_cast<T*>(nullptr);

  if (dx) dx_data = dev_ctx.template Alloc<T>(dx);
  if (dy) dy_data = dev_ctx.template Alloc<T>(dy);

  if (x_ndim == 1 && y_ndim == 1) {
    if (out_grad.numel() == 1) {
      auto M = 1;
      auto K = 1;
      auto N = x.numel();
      if (dx) {
        GEMM(false, false, M, K, N, out_grad_data, y_data, dx_data);
      }
      if (dy) {
        GEMM(false, false, M, K, N, out_grad_data, x_data, dy_data);
      }
      return;
    }
  } else if (x_ndim == 1) {
    // dx = dout' * y
    // dy = x * dout'
    if (dx) {
      auto M = 1;
      auto K = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
      auto N = transpose_y ? y_dims[y_ndim - 1] : y_dims[y_ndim - 2];
      if (y_ndim == 2) {
        GEMM(false, !transpose_y, M, K, N, out_grad_data, y_data, dx_data);
      } else {
        BatchedGEMM(false,
                    !transpose_y,
                    M,
                    K,
                    N,
                    out_grad_data,
                    y_data,
                    dx_data,
                    y_dims[0],
                    false,
                    false,
                    true,
                    true);
      }
    }
    if (dy) {
      auto M = transpose_y ? y_dims[y_ndim - 1] : y_dims[y_ndim - 2];
      auto K = 1;
      auto N = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
      if (y_ndim == 2) {
        GEMM(
            false, false, M, K, N, x_data, out_grad_data, dy_data, transpose_y);
      } else {
        BatchedGEMM(false,
                    false,
                    M,
                    K,
                    N,
                    x_data,
                    out_grad_data,
                    dy_data,
                    y_dims[0],
                    false,
                    transpose_y);
      }
    }
    return;
  } else if (y_ndim == 1) {
    // dx = dout * y'
    if (dx) {
      auto M = transpose_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
      auto K = 1;
      auto N = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
      if (x_ndim == 2) {
        GEMM(
            false, false, M, K, N, out_grad_data, y_data, dx_data, transpose_x);
      } else {
        BatchedGEMM(false,
                    false,
                    M,
                    K,
                    N,
                    out_grad_data,
                    y_data,
                    dx_data,
                    x_dims[0],
                    true,
                    transpose_x);
      }
    }
    // dy = x' * dout
    if (dy) {
      auto M = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
      auto K = transpose_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
      auto N = 1;
      if (x_ndim == 2) {
        GEMM(!transpose_x, false, M, K, N, x_data, out_grad_data, dy_data);
      } else {
        BatchedGEMM(!transpose_x,
                    false,
                    M,
                    K,
                    N,
                    x_data,
                    out_grad_data,
                    dy_data,
                    x_dims[0],
                    true,
                    false,
                    true,
                    true);
      }
    }

    return;
  }

  if (x_ndim == 2 && y_ndim == 2) {
    // dx = dout * y'
    if (dx) {
      auto M = transpose_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
      auto K = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
      auto N = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
      GEMM(false,
           !transpose_y,
           M,
           K,
           N,
           out_grad_data,
           y_data,
           dx_data,
           transpose_x);
    }
    // dy = x' * dout
    if (dy) {
      auto M = transpose_y ? y_dims[y_ndim - 1] : y_dims[y_ndim - 2];
      auto K = transpose_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
      auto N = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
      GEMM(!transpose_x,
           false,
           M,
           K,
           N,
           x_data,
           out_grad_data,
           dy_data,
           transpose_y);
    }
    return;
  } else if (x_ndim == 2) {
    // dx = dout * y'
    if (dx) {
      auto M = transpose_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
      auto K = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
      auto N = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
      BatchedGEMM(false,
                  !transpose_y,
                  M,
                  K,
                  N,
                  out_grad_data,
                  y_data,
                  dx_data,
                  y_dims[0],
                  false,
                  transpose_x,
                  true,
                  true);
    }
    // dy = x' * dout
    if (dy) {
      auto M = transpose_y ? y_dims[y_ndim - 1] : y_dims[y_ndim - 2];
      auto K = transpose_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
      auto N = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
      BatchedGEMM(!transpose_x,
                  false,
                  M,
                  K,
                  N,
                  x_data,
                  out_grad_data,
                  dy_data,
                  y_dims[0],
                  false,
                  transpose_y);
    }
    return;
  } else if (y_ndim == 2) {
    // dx = dout * y'
    if (dx) {
      auto M = transpose_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
      auto K = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
      auto N = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
      BatchedGEMM(false,
                  !transpose_y,
                  M,
                  K,
                  N,
                  out_grad_data,
                  y_data,
                  dx_data,
                  x_dims[0],
                  true,
                  transpose_x);
    }
    // dy = x' * dout
    if (dy) {
      auto M = transpose_y ? y_dims[y_ndim - 1] : y_dims[y_ndim - 2];
      auto K = transpose_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
      auto N = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
      GEMM(!transpose_x,
           false,
           M,
           K * x_dims[0],
           N,
           x_data,
           out_grad_data,
           dy_data,
           transpose_y);
    }
    return;
  }
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(matmul,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::MatmulKernel,
                    phi::dtype::float16,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(matmul_grad,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::MatmulGradKernel,
                    phi::dtype::float16,
                    float,
                    double) {}
