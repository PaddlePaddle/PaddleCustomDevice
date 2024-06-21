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

#include <iostream>
#include <typeinfo>
#include <vector>

#include "kernels/funcs/tblas_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void MatmulKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA MatmulKernel";

  dev_ctx.template Alloc<T>(out);

  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  std::vector<int64_t> out_dims = phi::vectorize(out->dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int out_ndim = out_dims.size();

  // Case 1: [K] x [K] = [1]
  if (x_ndim == 1 && y_ndim == 1) {
    PADDLE_ENFORCE_EQ(
        x.numel(),
        y.numel(),
        phi::errors::InvalidArgument(
            "X's numbers must be equal to Y's numbers,"
            "when X/Y's dims =1. But received X has [%d] elements,"
            "received Y has [%d] elements",
            x.numel(),
            y.numel()));
    out->Resize({1});
    tblas_ops::Dot<T>(dev_ctx, x, y, out);
    return;
  }

  // Case 2: [M, K] x [K] = [M]
  if (x_ndim == 2 && y_ndim == 1) {
    if (transpose_x) {
      PADDLE_ENFORCE_EQ(x_dims[x_ndim - 2],
                        y.numel(),
                        phi::errors::InvalidArgument(
                            "X'dims[%d] must be equal to Y's element numbers."
                            "But received X'dims[%d] is %d,"
                            "received Y has [%d] elements.",
                            x_ndim - 2,
                            x_ndim - 2,
                            x_dims[x_ndim - 2],
                            y.numel()));
    } else {
      PADDLE_ENFORCE_EQ(x_dims[x_ndim - 1],
                        y.numel(),
                        phi::errors::InvalidArgument(
                            "X'dims[%d] must be equal to Y's element numbers."
                            "But received X'dims[%d] is %d,"
                            "received Y has [%d] elements.",
                            x_ndim - 1,
                            x_ndim - 1,
                            x_dims[x_ndim - 1],
                            y.numel()));
    }
    tblas_ops::MatVec<T>(dev_ctx, x, y, transpose_x, out);
    return;
  }

  // Resize dim 1 to 2
  phi::DenseTensor x_temp(x), y_temp(y);
  if (x_ndim == 1) {
    x_dims.insert(x_dims.begin(), 1);
    out_dims.insert(out_dims.end() - 1, 1);
    x_temp.Resize(phi::make_ddim(x_dims));
    x_ndim = 2;
    out_ndim += 1;
  }
  if (y_ndim == 1) {
    y_dims.push_back(1);
    out_dims.push_back(1);
    y_temp.Resize(phi::make_ddim(y_dims));
    y_ndim = 2;
    out_ndim += 1;
  }

  const int K = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (transpose_y) {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 1],
        K,
        phi::errors::InvalidArgument("Input(Y) has error dim."
                                     "Y'dims[%d] must be equal to %d"
                                     "But received Y'dims[%d] is %d",
                                     y_ndim - 1,
                                     K,
                                     y_ndim - 1,
                                     y_dims[y_ndim - 1]));
  } else {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 2],
        K,
        phi::errors::InvalidArgument("Input(Y) has error dim."
                                     "Y'dims[%d] must be equal to %d"
                                     "But received Y'dims[%d] is %d",
                                     y_ndim - 2,
                                     K,
                                     y_ndim - 2,
                                     y_dims[y_ndim - 2]));
  }

  // Case 3: [M, K] x [K, N] = [M, N]
  if (x_ndim == 2 && y_ndim == 2) {
    tblas_ops::MatMul2D<T>(
        dev_ctx, x_temp, y_temp, transpose_x, transpose_y, out);
    return;
  }

  // Case 4: [A, B, ..., M, K] x [K, N] = [A, B, ..., M, N]
  // Case 4_1: [M, K] x [A, B, ..., K, N] = [A, B, ..., M, N]
  if (x_ndim > 2 && y_ndim == 2) {
    tblas_ops::BatchedMatmulWithSingleMat<T>(
        dev_ctx, x_temp, y_temp, transpose_x, transpose_y, out);
    return;
  } else if (x_ndim == 2 && y_ndim > 2) {
    tblas_ops::SingleMatmulWithBatchedMat<T>(
        dev_ctx, x_temp, y_temp, transpose_x, transpose_y, out);
    return;
  }

  // Case 5: [B, M, K] x [B, K, N] = [B, M, N]
  if (x_ndim > 2 && y_ndim > 2) {
    auto x_non_mat_dims =
        phi::slice_ddim(x_temp.dims(), 0, x_temp.dims().size() - 2);
    auto y_non_mat_dims =
        phi::slice_ddim(y_temp.dims(), 0, y_temp.dims().size() - 2);
    if (x_non_mat_dims == y_non_mat_dims) {
      tblas_ops::BatchMatmul<T>(dev_ctx, x, y, transpose_x, transpose_y, out);
      return;
    }
  }

  // Case 6: Bidirectional Broadcast
  std::vector<int64_t> x_broadcast_dims(out_ndim, 1);
  std::vector<int64_t> y_broadcast_dims(out_ndim, 1);
  std::copy(out_dims.begin(), out_dims.end() - 2, x_broadcast_dims.begin());
  std::copy(out_dims.begin(), out_dims.end() - 2, y_broadcast_dims.begin());
  std::copy(x_dims.end() - 2, x_dims.end(), x_broadcast_dims.end() - 2);
  std::copy(y_dims.end() - 2, y_dims.end(), y_broadcast_dims.end() - 2);

  int x_dims_size = x_dims.size();
  if (x_broadcast_dims.size() > x_dims_size) {
    for (int i = 0; i < (x_broadcast_dims.size() - x_dims_size); i++) {
      x_dims.insert(x_dims.begin(), 1);
      x_ndim += 1;
    }
  }

  int y_dims_size = y_dims.size();
  if (y_broadcast_dims.size() > y_dims_size) {
    for (int j = 0; j < (y_broadcast_dims.size() - y_dims_size); j++) {
      y_dims.insert(y_dims.begin(), 1);
      y_ndim += 1;
    }
  }

  std::vector<T*> a, b, c;
  std::vector<int64_t> a_mat_dims = {x_dims[x_dims.size() - 2],
                                     x_dims[x_dims.size() - 1]};
  std::vector<int64_t> b_mat_dims = {y_dims[y_dims.size() - 2],
                                     y_dims[y_dims.size() - 1]};
  T* x_ptr = const_cast<T*>(x.data<T>());
  T* y_ptr = const_cast<T*>(y.data<T>());
  T* out_ptr = out->data<T>();

  tblas_ops::doBroadcastTo<T>(
      x_ptr, y_ptr, out_ptr, &x_dims, &y_dims, &out_dims, &a, &b, &c);
  tblas_ops::MatMulND<T>(
      dev_ctx, a, a_mat_dims, b, b_mat_dims, c, transpose_x, transpose_y, out);
}

template <typename T, typename Context>
void MatmulGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      const phi::DenseTensor& dout,
                      bool transpose_x,
                      bool transpose_y,
                      phi::DenseTensor* dx,
                      phi::DenseTensor* dy) {
  VLOG(4) << "Call SDAA MatmulGradKernel";
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
  }

  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  std::vector<int64_t> dout_dims = phi::vectorize(dout.dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int dout_ndim = dout_dims.size();
  std::vector<int64_t> x_inital_dims = x_dims;
  std::vector<int64_t> y_inital_dims = y_dims;
  std::vector<int64_t> dout_inital_dims = dout_dims;
  T* x_ptr = const_cast<T*>(x.data<T>());
  T* y_ptr = const_cast<T*>(y.data<T>());
  T* dout_ptr = const_cast<T*>(dout.data<T>());

  // Case 1: [K] x [K] = [1]
  if (x_ndim == 1 && y_ndim == 1) {
    if (dx) {
      tblas_ops::DotGradFunction<T>(dev_ctx, dout, y, dx);
    }
    if (dy) {
      tblas_ops::DotGradFunction<T>(dev_ctx, dout, x, dy);
    }
    return;
  }

  // Resize dim 1 to 2
  phi::DenseTensor x_temp(x), y_temp(y), dout_temp(dout), dx_temp, dy_temp;
  if (dx) {
    dx_temp = *dx;
  }

  if (dy) {
    dy_temp = *dy;
  }

  if (x_ndim == 1) {
    x_dims.insert(x_dims.begin(), 1);
    dout_dims.insert(dout_dims.end() - 1, 1);
    x_temp.Resize(phi::make_ddim(x_dims));
    dx_temp.Resize(phi::make_ddim(x_dims));
    dout_temp.Resize(phi::make_ddim(dout_dims));
    x_ndim = 2;
    dout_ndim += 1;
  }
  if (y_ndim == 1) {
    y_dims.push_back(1);
    dout_dims.push_back(1);
    y_temp.Resize(phi::make_ddim(y_dims));
    dy_temp.Resize(phi::make_ddim(y_dims));
    dout_temp.Resize(phi::make_ddim(dout_dims));
    y_ndim = 2;
    dout_ndim += 1;
  }

  // Case 2: [M, K] x [K] = [M], merge to case 3.
  // Case 3: [M, k] x [K, N] = [M, N]
  if (x_ndim == 2 && y_ndim == 2) {
    if (dx) {
      dx->Resize(phi::make_ddim(x_dims));
      if (transpose_x) {
        tblas_ops::MatMul2D<T>(
            dev_ctx, y_temp, dout_temp, transpose_y, true, dx);
      } else {
        tblas_ops::MatMul2D<T>(
            dev_ctx, dout_temp, y_temp, false, !transpose_y, dx);
      }
      dx->Resize(x.dims());
    }
    if (dy) {
      dy->Resize(phi::make_ddim(y_dims));
      if (transpose_y) {
        tblas_ops::MatMul2D<T>(
            dev_ctx, dout_temp, x_temp, true, transpose_x, dy);
      } else {
        tblas_ops::MatMul2D<T>(
            dev_ctx, x_temp, dout_temp, !transpose_x, false, dy);
      }
      dy->Resize(y.dims());
    }
    return;
  }

  // Case 4: [A, B, ..., M, K] x [K, N] = [A, B, ..., M, N]
  if (x_ndim > 2 && y_ndim == 2) {
    if (dx) {
      if (transpose_x) {
        tblas_ops::SingleMatmulWithBatchedMat<T>(
            dev_ctx, y_temp, dout_temp, transpose_y, true, &dx_temp);
      } else {
        tblas_ops::BatchedMatmulWithSingleMat<T>(
            dev_ctx, dout_temp, y_temp, false, !transpose_y, &dx_temp);
      }
    }
    if (dy) {
      if (!transpose_x) {
        // [K, AB...M] x [AB...M, N] = [K, N]
        auto k = x_dims[x_ndim - 1];
        auto m = x_temp.numel() / k;
        auto n = dout_temp.numel() / m;
        x_temp.Resize({m, k});
        dout_temp.Resize({m, n});
        auto first_temp = transpose_y ? dout_temp : x_temp;
        auto second_temp = transpose_y ? x_temp : dout_temp;
        // trans_a and trans_b is fixed in this situation.
        tblas_ops::MatMul2D<T>(
            dev_ctx, first_temp, second_temp, true, false, &dy_temp);
      } else {
        // 1. [batch, K, M] x [batch, M, N] = [batch, K, N]
        // 2. [batch, K, N] --> [K, N]
        phi::DenseTensor dy_unreduced;
        auto batch = x_temp.numel() / x_dims[x_ndim - 1] / x_dims[x_ndim - 2];
        x_temp.Resize({batch, x_dims[x_ndim - 2], x_dims[x_ndim - 1]});
        dout_temp.Resize(
            {batch, dout_dims[dout_ndim - 2], dout_dims[dout_ndim - 1]});
        dy_unreduced.Resize({batch, y_dims[0], y_dims[1]});
        dev_ctx.template Alloc<T>(&dy_unreduced);
        auto first_temp = transpose_y ? dout_temp : x_temp;
        auto second_temp = transpose_y ? x_temp : dout_temp;
        auto trans_a = transpose_y ? true : !transpose_x;
        auto trans_b = transpose_y ? transpose_x : false;
        tblas_ops::BatchMatmul<T>(
            dev_ctx, first_temp, second_temp, trans_a, trans_b, &dy_unreduced);
        sdaa_ops::doSumTensor(dev_ctx, dy_unreduced, {0}, &dy_temp);
      }
    }
    return;
  }

  // Case 4_1: [M, K] x [A, B, ..., K, N] = [A, B, ..., M, N]
  if (x_ndim == 2 && y_ndim > 2) {
    // 1. [batch, M, N] x [batch, N, K] = [batch, M, K]
    // 2. [batch, M, K] --> [M, K]
    if (dx) {
      phi::DenseTensor dx_unreduced;
      auto batch = y_temp.numel() / y_dims[y_ndim - 1] / y_dims[y_ndim - 2];
      y_temp.Resize({batch, y_dims[y_ndim - 2], y_dims[y_ndim - 1]});
      dout_temp.Resize(
          {batch, dout_dims[dout_ndim - 2], dout_dims[dout_ndim - 1]});
      dx_unreduced.Resize({batch, x_dims[0], x_dims[1]});
      dev_ctx.template Alloc<T>(&dx_unreduced);
      auto first_temp = transpose_x ? y_temp : dout_temp;
      auto second_temp = transpose_x ? dout_temp : y_temp;
      auto trans_a = transpose_x ? transpose_y : false;
      auto trans_b = transpose_x ? true : !transpose_y;
      tblas_ops::BatchMatmul<T>(
          dev_ctx, first_temp, second_temp, trans_a, trans_b, &dx_unreduced);
      sdaa_ops::doSumTensor(dev_ctx, dx_unreduced, {0}, &dx_temp);
    }
    if (dy) {
      if (transpose_y) {
        tblas_ops::BatchedMatmulWithSingleMat<T>(
            dev_ctx, dout_temp, x_temp, true, transpose_x, &dy_temp);
      } else {
        tblas_ops::SingleMatmulWithBatchedMat<T>(
            dev_ctx, x_temp, dout_temp, !transpose_x, false, &dy_temp);
      }
    }
    return;
  }

  auto x_non_mat_dims =
      phi::slice_ddim(x_temp.dims(), 0, x_temp.dims().size() - 2);
  auto y_non_mat_dims =
      phi::slice_ddim(y_temp.dims(), 0, y_temp.dims().size() - 2);

  // Case 5: [B, M, K] x [B, K, N] = [B, M, N]
  if (x_ndim > 2 && y_ndim > 2 && x_non_mat_dims == y_non_mat_dims) {
    if (dx) {
      if (transpose_x) {
        tblas_ops::BatchMatmul<T>(
            dev_ctx, y_temp, dout_temp, transpose_y, true, &dx_temp);
      } else {
        tblas_ops::BatchMatmul<T>(
            dev_ctx, dout_temp, y_temp, false, !transpose_y, &dx_temp);
      }
    }
    if (dy) {
      if (transpose_y) {
        tblas_ops::BatchMatmul<T>(
            dev_ctx, dout_temp, x_temp, true, transpose_x, &dy_temp);
      } else {
        tblas_ops::BatchMatmul<T>(
            dev_ctx, x_temp, dout_temp, !transpose_x, false, &dy_temp);
      }
    }
    return;
  }

  const int K = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  const int N = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];

  // Case 6: Bidirectional Broadcast
  std::vector<int64_t> x_broadcast_dims(dout_ndim, 1);
  std::vector<int64_t> y_broadcast_dims(dout_ndim, 1);
  std::copy(dout_dims.begin(), dout_dims.end() - 2, x_broadcast_dims.begin());
  std::copy(dout_dims.begin(), dout_dims.end() - 2, y_broadcast_dims.begin());
  std::copy(x_dims.end() - 2, x_dims.end(), x_broadcast_dims.end() - 2);
  std::copy(y_dims.end() - 2, y_dims.end(), y_broadcast_dims.end() - 2);

  int x_dims_size = x_dims.size();
  if (x_broadcast_dims.size() > x_dims_size) {
    for (int i = 0; i < (x_broadcast_dims.size() - x_dims_size); i++) {
      x_dims.insert(x_dims.begin(), 1);
      x_ndim += 1;
    }
  }

  int y_dims_size = y_dims.size();
  if (y_broadcast_dims.size() > y_dims_size) {
    for (int j = 0; j < (y_broadcast_dims.size() - y_dims_size); j++) {
      y_dims.insert(y_dims.begin(), 1);
      y_ndim += 1;
    }
  }

  // allocate memory for dx_braodcast if x_dims != x_broadcast_dims
  size_t dx_brd_size = 1;
  for (int idx = 0; idx < x_broadcast_dims.size(); idx++) {
    dx_brd_size *= x_broadcast_dims[idx];
  }
  phi::DenseTensor dx_brd;
  phi::DDim dx_brd_dims = phi::make_ddim(x_broadcast_dims);
  phi::DenseTensorMeta dx_meta = {dout.dtype(), dx_brd_dims};
  dx_brd.set_meta(dx_meta);
  dev_ctx.template Alloc<T>(&dx_brd);

  // allocate memory for dy_braodcast if y_dims != y_broadcast_dims
  size_t dy_brd_size = 1;
  for (int idx = 0; idx < y_broadcast_dims.size(); idx++) {
    dy_brd_size *= y_broadcast_dims[idx];
  }
  phi::DenseTensor dy_brd;
  phi::DDim dy_brd_dims = phi::make_ddim(y_broadcast_dims);
  phi::DenseTensorMeta dy_meta = {dout.dtype(), dy_brd_dims};
  dy_brd.set_meta(dy_meta);
  dev_ctx.template Alloc<T>(&dy_brd);

  std::vector<T*> x_brd_address, y_brd_address, dout_address_dx,
      dout_address_dy, dx_address, dy_address;
  std::vector<int64_t> x_a_mat_dims = {x_dims[x_dims.size() - 2],
                                       x_dims[x_dims.size() - 1]};
  std::vector<int64_t> y_a_mat_dims = {y_dims[y_dims.size() - 2],
                                       y_dims[y_dims.size() - 1]};
  std::vector<int64_t> dout_a_mat_dims = {dout_dims[dout_dims.size() - 2],
                                          dout_dims[dout_dims.size() - 1]};

  if (dx) {
    if (x_dims == x_broadcast_dims) {
      tblas_ops::doBroadcastTo<T>(y_ptr,
                                  dout_ptr,
                                  dx->data<T>(),
                                  &y_dims,
                                  &dout_dims,
                                  &x_dims,
                                  &y_brd_address,
                                  &dout_address_dx,
                                  &dx_address);
      if (transpose_x) {
        tblas_ops::MatMulND<T>(dev_ctx,
                               y_brd_address,
                               y_a_mat_dims,
                               dout_address_dx,
                               dout_a_mat_dims,
                               dx_address,
                               transpose_y,
                               true,
                               dx);
      } else {
        tblas_ops::MatMulND<T>(dev_ctx,
                               dout_address_dx,
                               dout_a_mat_dims,
                               y_brd_address,
                               y_a_mat_dims,
                               dx_address,
                               false,
                               !transpose_y,
                               dx);
      }
    } else {
      tblas_ops::doBroadcastTo<T>(y_ptr,
                                  dout_ptr,
                                  dx_brd.data<T>(),
                                  &y_dims,
                                  &dout_dims,
                                  &x_broadcast_dims,
                                  &y_brd_address,
                                  &dout_address_dx,
                                  &dx_address);
      if (transpose_x) {
        tblas_ops::MatMulND<T>(dev_ctx,
                               y_brd_address,
                               y_a_mat_dims,
                               dout_address_dx,
                               dout_a_mat_dims,
                               dx_address,
                               transpose_y,
                               true,
                               &dx_brd);
      } else {
        tblas_ops::MatMulND<T>(dev_ctx,
                               dout_address_dx,
                               dout_a_mat_dims,
                               y_brd_address,
                               y_a_mat_dims,
                               dx_address,
                               false,
                               !transpose_y,
                               &dx_brd);
      }
      // need to reduce dx_brd dimension to dx dimension if x_dims !=
      // x_broadcast_dims
      std::vector<int64_t> dx_reduce_dims;
      assert(x_dims.size() == x_broadcast_dims.size());
      for (int i = 0; i < x_dims.size(); i++) {
        if (x_dims[i] != x_broadcast_dims[i]) {
          dx_reduce_dims.push_back(i);
        }
      }
      sdaa_ops::doSumTensor(dev_ctx, dx_brd, dx_reduce_dims, dx);
    }
  }
  if (dy) {
    if (y_dims == y_broadcast_dims) {
      tblas_ops::doBroadcastTo<T>(x_ptr,
                                  dout_ptr,
                                  dy->data<T>(),
                                  &x_dims,
                                  &dout_dims,
                                  &y_dims,
                                  &x_brd_address,
                                  &dout_address_dy,
                                  &dy_address);
      if (transpose_y) {
        tblas_ops::MatMulND<T>(dev_ctx,
                               dout_address_dy,
                               dout_a_mat_dims,
                               x_brd_address,
                               x_a_mat_dims,
                               dy_address,
                               true,
                               transpose_x,
                               dy);
      } else {
        tblas_ops::MatMulND<T>(dev_ctx,
                               x_brd_address,
                               x_a_mat_dims,
                               dout_address_dy,
                               dout_a_mat_dims,
                               dy_address,
                               !transpose_x,
                               false,
                               dy);
      }
    } else {
      tblas_ops::doBroadcastTo<T>(x_ptr,
                                  dout_ptr,
                                  dy_brd.data<T>(),
                                  &x_dims,
                                  &dout_dims,
                                  &y_broadcast_dims,
                                  &x_brd_address,
                                  &dout_address_dy,
                                  &dy_address);
      if (transpose_y) {
        tblas_ops::MatMulND<T>(dev_ctx,
                               dout_address_dy,
                               dout_a_mat_dims,
                               x_brd_address,
                               x_a_mat_dims,
                               dy_address,
                               true,
                               transpose_x,
                               &dy_brd);
      } else {
        tblas_ops::MatMulND<T>(dev_ctx,
                               x_brd_address,
                               x_a_mat_dims,
                               dout_address_dy,
                               dout_a_mat_dims,
                               dy_address,
                               !transpose_x,
                               false,
                               &dy_brd);
      }
      // need to reduce dy_brd dimension to dy dimension if y_dims !=
      // y_broadcast_dims
      std::vector<int64_t> dy_reduce_dims;
      assert(y_dims.size() == y_broadcast_dims.size());
      for (int i = 0; i < y_dims.size(); i++) {
        if (y_dims[i] != y_broadcast_dims[i]) {
          dy_reduce_dims.push_back(i);
        }
      }
      sdaa_ops::doSumTensor(dev_ctx, dy_brd, dy_reduce_dims, dy);
    }
  }
}

template <typename T, typename Context>
void MatmulWithFlattenKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& y,
                             int x_num_col_dims,
                             int y_num_col_dims,
                             phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA MatmulWithFlattenKernel";

  dev_ctx.template Alloc<T>(out);

  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();

  PADDLE_ENFORCE_GT(
      x_ndim,
      1,
      phi::errors::InvalidArgument("The x dims shoud be greater than 1."
                                   "But got %d.",
                                   x_ndim));

  PADDLE_ENFORCE_GT(
      y_ndim,
      1,
      phi::errors::InvalidArgument("The y dims shoud be greater than 1."
                                   "But got %d.",
                                   y_ndim));

  if (x_ndim == 2 && y_ndim == 2) {
    PADDLE_ENFORCE_EQ(
        x_dims[1],
        y_dims[0],
        phi::errors::InvalidArgument("Input(X) second dimension must be equal "
                                     "to Input(Y) first dimension."
                                     "But got Input(X) second dimension is %d."
                                     "Input(Y) fisrt dimension is %d.",
                                     x_dims[1],
                                     y_dims[0]));
    tblas_ops::MatMul2D<T>(dev_ctx, x, y, false, false, out);
  } else {
    // flatten
    std::vector<int64_t> x_matrix_dims, y_matrix_dims;

    tblas_ops::ReshpaeToMatrix(x_num_col_dims, x_dims, &x_matrix_dims);
    tblas_ops::ReshpaeToMatrix(y_num_col_dims, y_dims, &y_matrix_dims);

    PADDLE_ENFORCE_EQ(
        x_matrix_dims[1],
        y_matrix_dims[0],
        phi::errors::InvalidArgument(
            "Input(X_With_Flatten) second dimension must be equal to "
            "Input(Y_With_Flatten) first dimension."
            "But got Input(X_With_Flatten) second dimension is %d."
            "Input(Y_With_Flatten) fisrt dimension is %d.",
            x_matrix_dims[1],
            y_matrix_dims[0]));

    phi::DenseTensor x_matrix(x);
    x_matrix.Resize(phi::make_ddim(x_matrix_dims));
    phi::DenseTensor y_matrix(y);
    y_matrix.Resize(phi::make_ddim(y_matrix_dims));

    tblas_ops::MatMul2D<T>(dev_ctx, x_matrix, y_matrix, false, false, out);
  }
}

template <typename T, typename Context>
void MatmulWithFlattenGradKernel(const Context& dev_ctx,
                                 const phi::DenseTensor& x,
                                 const phi::DenseTensor& y,
                                 const phi::DenseTensor& dout,
                                 int x_num_col_dims,
                                 int y_num_col_dims,
                                 phi::DenseTensor* dx,
                                 phi::DenseTensor* dy) {
  VLOG(4) << "CALL SDAA MatmulWithFlattenGradKernel";

  if (dx) dev_ctx.template Alloc<T>(dx);
  if (dy) dev_ctx.template Alloc<T>(dy);

  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();

  PADDLE_ENFORCE_GT(
      x_ndim,
      1,
      phi::errors::InvalidArgument("The x dims shoud be greater than 1."
                                   "But got %d.",
                                   x_ndim));

  PADDLE_ENFORCE_GT(
      y_ndim,
      1,
      phi::errors::InvalidArgument("The y dims shoud be greater than 1."
                                   "But got %d.",
                                   y_ndim));

  if (x_ndim == 2 && y_ndim == 2) {
    if (dx) {
      tblas_ops::MatMul2D<T>(dev_ctx, dout, y, false, true, dx);
    }
    if (dy) {
      tblas_ops::MatMul2D<T>(dev_ctx, x, dout, true, false, dy);
    }
  } else {
    // flatten
    std::vector<int64_t> x_matrix_dims, y_matrix_dims;

    tblas_ops::ReshpaeToMatrix(x_num_col_dims, x_dims, &x_matrix_dims);
    tblas_ops::ReshpaeToMatrix(y_num_col_dims, y_dims, &y_matrix_dims);
    std::vector<int64_t> dout_temp_dims = {x_matrix_dims[0], y_matrix_dims[1]};

    phi::DenseTensor x_matrix(x);
    x_matrix.Resize(phi::make_ddim(x_matrix_dims));
    phi::DenseTensor y_matrix(y);
    y_matrix.Resize(phi::make_ddim(y_matrix_dims));
    phi::DenseTensor dout_temp(dout);
    dout_temp.Resize(phi::make_ddim(dout_temp_dims));

    if (dx) {
      tblas_ops::MatMul2D<T>(dev_ctx, dout_temp, y_matrix, false, true, dx);
    }
    if (dy) {
      tblas_ops::MatMul2D<T>(dev_ctx, x_matrix, dout_temp, true, false, dy);
    }
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(matmul,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MatmulKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(matmul_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MatmulGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(matmul_with_flatten,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MatmulWithFlattenKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(matmul_with_flatten_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MatmulWithFlattenGradKernel,
                          float,
                          phi::dtype::float16) {}
