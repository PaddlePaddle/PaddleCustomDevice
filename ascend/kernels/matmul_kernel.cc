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

#include "npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
static void MatMul2D(const Context& dev_ctx,
                     const aclrtStream& stream,
                     const phi::DenseTensor& X,
                     const phi::DenseTensor& Y,
                     phi::DenseTensor* out,
                     const bool transpose_x,
                     const bool transpose_y) {
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner(
      "MatMul",
      {X, Y},
      {*out},
      {{"transpose_x1", transpose_x}, {"transpose_x2", transpose_y}});
  runner.Run(stream);
}

template <typename T, typename Context>
static void MatMulND(const Context& dev_ctx,
                     const aclrtStream& stream,
                     const phi::DenseTensor& X,
                     const phi::DenseTensor& Y,
                     phi::DenseTensor* out,
                     const bool transpose_x,
                     const bool transpose_y) {
  dev_ctx.template Alloc<T>(out);
  const auto& runner =
      NpuOpRunner("BatchMatMul",
                  {X, Y},
                  {*out},
                  {{"adj_x1", transpose_x}, {"adj_x2", transpose_y}});
  runner.Run(stream);
}

template <typename T, typename Context>
static void ReduceDims(const Context& dev_ctx,
                       const aclrtStream& stream,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& brd_dims,
                       const phi::DenseTensor& in,
                       phi::DenseTensor* out) {
  std::vector<int64_t> axes;
  int64_t size = brd_dims.size();
  int64_t diff = brd_dims.size() - dims.size();
  for (int64_t i = 0; i < size; ++i) {
    if (i < diff) {
      axes.push_back(i);
      continue;
    }
    if (brd_dims[i] > dims[i - diff]) {
      axes.push_back(i);
    }
  }
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner(
      "ReduceSumD", {in}, {*out}, {{"axes", axes}, {"keep_dims", false}});
  runner.Run(stream);
}

template <typename T, typename Context>
void MatmulKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  phi::DenseTensor* out) {
  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  std::vector<int64_t> out_dims = phi::vectorize(out->dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int out_ndim = out_dims.size();

  auto stream = dev_ctx.stream();

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
    dev_ctx.template Alloc<T>(out);

    const auto& runner = NpuOpRunner("Dot", {x, y}, {*out});
    runner.Run(stream);
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

  // Case 2: [M, K] x [K, N] = [M, N]
  if (x_ndim == 2 && y_ndim == 2) {
    MatMul2D<T>(dev_ctx, stream, x_temp, y_temp, out, transpose_x, transpose_y);
    return;
  }

  // Case 3: [B, M, K] x [K, N] =  [B, M, N], when transpose_x = false
  // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
  if (transpose_x == false && y_ndim == 2) {
    std::vector<int64_t> vec_dim = {x_temp.numel() / K, K};
    x_temp.Resize(phi::make_ddim(vec_dim));
    MatMul2D<T>(dev_ctx, stream, x_temp, y_temp, out, transpose_x, transpose_y);
    return;
  }

  // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
  std::vector<int64_t> x_broadcast_dims(out_ndim, 1);
  std::vector<int64_t> y_broadcast_dims(out_ndim, 1);
  std::copy(out_dims.begin(), out_dims.end() - 2, x_broadcast_dims.begin());
  std::copy(out_dims.begin(), out_dims.end() - 2, y_broadcast_dims.begin());
  std::copy(x_dims.end() - 2, x_dims.end(), x_broadcast_dims.end() - 2);
  std::copy(y_dims.end() - 2, y_dims.end(), y_broadcast_dims.end() - 2);

  phi::DenseTensor x_temp_brd;
  phi::DenseTensorMeta x_temp_brd_meta = {x.dtype(), {}};
  x_temp_brd.set_meta(x_temp_brd_meta);
  if (x_dims == x_broadcast_dims) {
    x_temp_brd = x;
    x_temp_brd.Resize(phi::make_ddim(x_broadcast_dims));
  } else {
    x_temp_brd.Resize(phi::make_ddim(x_broadcast_dims));
    dev_ctx.template Alloc<T>(&x_temp_brd);
    NpuOpRunner runner_brd;
    runner_brd.SetType("BroadcastTo")
        .AddInput(x_temp)
        .AddInput(dev_ctx, std::move(x_broadcast_dims))
        .AddOutput(x_temp_brd)
        .Run(stream);
  }

  phi::DenseTensor y_temp_brd;
  phi::DenseTensorMeta y_temp_brd_meta = {y.dtype(), {}};
  y_temp_brd.set_meta(y_temp_brd_meta);
  if (y_dims == y_broadcast_dims) {
    y_temp_brd = y;
    y_temp_brd.Resize(phi::make_ddim(y_broadcast_dims));
  } else {
    y_temp_brd.Resize(phi::make_ddim(y_broadcast_dims));
    dev_ctx.template Alloc<T>(&y_temp_brd);
    NpuOpRunner runner_brd;
    runner_brd.SetType("BroadcastTo")
        .AddInput(y_temp)
        .AddInput(dev_ctx, std::move(y_broadcast_dims))
        .AddOutput(y_temp_brd)
        .Run(stream);
  }
  MatMulND<T>(
      dev_ctx, stream, x_temp_brd, y_temp_brd, out, transpose_x, transpose_y);
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
  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  std::vector<int64_t> out_dims = phi::vectorize(dout.dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int out_ndim = out_dims.size();

  auto stream = dev_ctx.stream();

  // Case 1: [K] x [K] = [1]
  if (x_ndim == 1 && y_ndim == 1) {
    phi::DenseTensor dout_temp;
    phi::DenseTensorMeta dout_temp_meta = {dout.dtype(), x.dims()};
    dout_temp.set_meta(dout_temp_meta);
    dev_ctx.template Alloc<T>(&dout_temp);
    NpuOpRunner runner;
    runner.SetType("BroadcastTo")
        .AddInput(dout)
        .AddInput(dev_ctx, std::move(x_dims))
        .AddOutput(dout_temp)
        .Run(stream);

    if (dx) {
      dev_ctx.template Alloc<T>(dx);
      const auto& runner_dx = NpuOpRunner("Mul", {dout_temp, y}, {*dx}, {});
      runner_dx.Run(stream);
    }
    if (dy) {
      dev_ctx.template Alloc<T>(dy);
      const auto& runner_dy = NpuOpRunner("Mul", {dout_temp, x}, {*dy}, {});
      runner_dy.Run(stream);
    }
    return;
  }

  // Resize dim 1 to 2
  phi::DenseTensor x_temp(x), y_temp(y), dout_temp(dout);
  if (x_ndim == 1) {
    x_dims.insert(x_dims.begin(), 1);
    out_dims.insert(out_dims.end() - 1, 1);
    x_temp.Resize(phi::make_ddim(x_dims));
    dout_temp.Resize(phi::make_ddim(out_dims));
    x_ndim = 2;
    out_ndim += 1;
  }
  if (y_ndim == 1) {
    y_dims.push_back(1);
    out_dims.push_back(1);
    y_temp.Resize(phi::make_ddim(y_dims));
    dout_temp.Resize(phi::make_ddim(out_dims));
    y_ndim = 2;
    out_ndim += 1;
  }

  // Case 2: [M, K] x [K, N] = [M, N]
  if (out_ndim == 2) {
    if (dx) {
      dx->Resize(phi::make_ddim(x_dims));
      if (transpose_x) {
        MatMul2D<T>(dev_ctx, stream, y_temp, dout_temp, dx, transpose_y, true);
      } else {
        MatMul2D<T>(
            dev_ctx, stream, dout_temp, y_temp, dx, false, !transpose_y);
      }
      dx->Resize(x.dims());
    }
    if (dy) {
      dy->Resize(phi::make_ddim(y_dims));
      if (transpose_y) {
        MatMul2D<T>(dev_ctx, stream, dout_temp, x_temp, dy, true, transpose_x);
      } else {
        MatMul2D<T>(
            dev_ctx, stream, x_temp, dout_temp, dy, !transpose_x, false);
      }
      dy->Resize(y.dims());
    }
    return;
  }

  const int K = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  const int N = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];

  // Case 3: [B, M, K] x [K, N] =  [B, M, N], when transpose_x = false
  // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
  if (transpose_x == false && y_ndim == 2) {
    std::vector<int64_t> x_vec_dim = {x_temp.numel() / K, K};
    dout_temp.Resize(
        phi::make_ddim(std::vector<int64_t>{dout_temp.numel() / N, N}));
    if (dx) {
      dx->Resize(phi::make_ddim(x_vec_dim));
      MatMul2D<T>(dev_ctx, stream, dout_temp, y_temp, dx, false, !transpose_y);
      dx->Resize(x.dims());
    }
    if (dy) {
      x_temp.Resize(phi::make_ddim(x_vec_dim));
      if (transpose_y) {
        MatMul2D<T>(dev_ctx, stream, dout_temp, x_temp, dy, true, false);
      } else {
        MatMul2D<T>(dev_ctx, stream, x_temp, dout_temp, dy, true, false);
      }
    }
    return;
  }

  // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
  std::vector<int64_t> x_broadcast_dims(out_ndim, 1);
  std::vector<int64_t> y_broadcast_dims(out_ndim, 1);
  std::copy(out_dims.begin(), out_dims.end() - 2, x_broadcast_dims.begin());
  std::copy(out_dims.begin(), out_dims.end() - 2, y_broadcast_dims.begin());
  std::copy(x_dims.end() - 2, x_dims.end(), x_broadcast_dims.end() - 2);
  std::copy(y_dims.end() - 2, y_dims.end(), y_broadcast_dims.end() - 2);

  phi::DenseTensor x_temp_brd;
  phi::DenseTensorMeta x_temp_brd_meta = {x.dtype(), {}};
  x_temp_brd.set_meta(x_temp_brd_meta);
  if (x_dims == x_broadcast_dims) {
    x_temp_brd = x;
    x_temp_brd.Resize(phi::make_ddim(x_broadcast_dims));
  } else {
    x_temp_brd.Resize(phi::make_ddim(x_broadcast_dims));
    dev_ctx.template Alloc<T>(&x_temp_brd);
    NpuOpRunner runner_brd;
    runner_brd.SetType("BroadcastTo")
        .AddInput(x_temp)
        .AddInput(dev_ctx, std::move(x_broadcast_dims))
        .AddOutput(x_temp_brd)
        .Run(stream);
  }

  phi::DenseTensor y_temp_brd;
  phi::DenseTensorMeta y_temp_brd_meta = {y.dtype(), {}};
  y_temp_brd.set_meta(y_temp_brd_meta);
  if (y_dims == y_broadcast_dims) {
    y_temp_brd = y;
    y_temp_brd.Resize(phi::make_ddim(y_broadcast_dims));
  } else {
    y_temp_brd.Resize(phi::make_ddim(y_broadcast_dims));
    dev_ctx.template Alloc<T>(&y_temp_brd);
    NpuOpRunner runner_brd;
    runner_brd.SetType("BroadcastTo")
        .AddInput(y_temp)
        .AddInput(dev_ctx, std::move(y_broadcast_dims))
        .AddOutput(y_temp_brd)
        .Run(stream);
  }

  if (dx) {
    if (x_dims == x_broadcast_dims) {
      if (transpose_x) {
        MatMulND<T>(
            dev_ctx, stream, y_temp_brd, dout_temp, dx, transpose_y, true);
      } else {
        MatMulND<T>(
            dev_ctx, stream, dout_temp, y_temp_brd, dx, false, !transpose_y);
      }
    } else {
      phi::DenseTensor dx_temp;
      phi::DenseTensorMeta dx_temp_meta = {x.dtype(),
                                           phi::make_ddim(x_broadcast_dims)};
      dx_temp.set_meta(dx_temp_meta);
      if (transpose_x) {
        MatMulND<T>(dev_ctx,
                    stream,
                    y_temp_brd,
                    dout_temp,
                    &dx_temp,
                    transpose_y,
                    true);
      } else {
        MatMulND<T>(dev_ctx,
                    stream,
                    dout_temp,
                    y_temp_brd,
                    &dx_temp,
                    false,
                    !transpose_y);
      }
      ReduceDims<T>(dev_ctx, stream, x_dims, x_broadcast_dims, dx_temp, dx);
    }
  }
  if (dy) {
    if (y_dims == y_broadcast_dims) {
      if (transpose_y) {
        MatMulND<T>(
            dev_ctx, stream, dout_temp, x_temp_brd, dy, true, transpose_x);
      } else {
        MatMulND<T>(
            dev_ctx, stream, x_temp_brd, dout_temp, dy, !transpose_x, false);
      }
    } else {
      phi::DenseTensor dy_temp;
      phi::DenseTensorMeta dy_temp_meta = {y.dtype(),
                                           phi::make_ddim(y_broadcast_dims)};
      dy_temp.set_meta(dy_temp_meta);
      if (transpose_y) {
        MatMulND<T>(dev_ctx,
                    stream,
                    dout_temp,
                    x_temp_brd,
                    &dy_temp,
                    true,
                    transpose_x);
      } else {
        MatMulND<T>(dev_ctx,
                    stream,
                    x_temp_brd,
                    dout_temp,
                    &dy_temp,
                    !transpose_x,
                    false);
      }
      ReduceDims<T>(dev_ctx, stream, y_dims, y_broadcast_dims, dy_temp, dy);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(matmul,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MatmulKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(matmul_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MatmulGradKernel,
                          float,
                          phi::dtype::float16) {}
