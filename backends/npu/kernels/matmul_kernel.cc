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

#include "kernels/funcs/npu_op_runner.h"
#include "kernels/funcs/op_command.h"

namespace custom_kernel {

template <typename T, typename Context>
static void BMMBroadcastTo(const Context& dev_ctx,
                           const std::vector<int>& x_dims,
                           const std::vector<int>& x_broadcast_dims,
                           const phi::DenseTensor& x,
                           phi::DenseTensor* x_broadcast) {
  VLOG(10) << "BMMBroadcastTo: (" << phi::make_ddim(x_dims) << "), ("
           << phi::make_ddim(x_broadcast_dims) << ")";

  x_broadcast->Resize(phi::make_ddim(x_broadcast_dims));
  dev_ctx.template Alloc<T>(x_broadcast);

  phi::DenseTensor broadcast_shape;
  experimental::OpCommandHelper::VectorToHostTensor(
      dev_ctx, x_broadcast_dims, &broadcast_shape);
  experimental::OpCommand("BroadcastTo")
      .Input(x,
             experimental::TensorDescMaker("x", x)
                 .SetDataLayout(phi::DataLayout::ANY)
                 .SetDims(phi::make_ddim(x_dims)))
      .Input(broadcast_shape,
             experimental::TensorDescMaker("shape", broadcast_shape)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Output(*x_broadcast,
              experimental::TensorDescMaker("y", *x_broadcast)
                  .SetDataLayout(phi::DataLayout::ANY))
      .Run(dev_ctx);
}

template <typename T, typename Context>
static void BMMKernel(const Context& dev_ctx,
                      const std::vector<int>& x_dims,
                      const std::vector<int>& y_dims,
                      const std::vector<int>& out_dims,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      bool transpose_x,
                      bool transpose_y,
                      phi::DenseTensor* out) {
  VLOG(10) << "BMMKernel: "
           << "(" << phi::make_ddim(x_dims) << "), "
           << "(" << phi::make_ddim(y_dims) << "), "
           << "(" << phi::make_ddim(out_dims) << ")";

  dev_ctx.template Alloc<T>(out);

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int out_ndim = out_dims.size();

  auto x_batch_size = std::accumulate(
      x_dims.cbegin(), x_dims.cend() - 2, 1, std::multiplies<int>());
  auto y_batch_size = std::accumulate(
      y_dims.cbegin(), y_dims.cend() - 2, 1, std::multiplies<int>());
  auto batch_size = std::accumulate(
      out_dims.cbegin(), out_dims.cend() - 2, 1, std::multiplies<int>());

  phi::DenseTensor x_broadcast, y_broadcast;
  if (x_batch_size != batch_size) {
    std::vector<int> broadcast_shape_vec(out_dims.cbegin(),
                                         out_dims.cend() - 2);
    broadcast_shape_vec.push_back(x_dims[x_ndim - 2]);
    broadcast_shape_vec.push_back(x_dims[x_ndim - 1]);
    BMMBroadcastTo<T, Context>(
        dev_ctx, x_dims, broadcast_shape_vec, x, &x_broadcast);
  } else {
    experimental::OpCommandHelper::Assign(dev_ctx, x, &x_broadcast);
  }

  if (y_batch_size != batch_size) {
    std::vector<int> broadcast_shape_vec(out_dims.cbegin(),
                                         out_dims.cend() - 2);
    broadcast_shape_vec.push_back(y_dims[y_ndim - 2]);
    broadcast_shape_vec.push_back(y_dims[y_ndim - 1]);
    BMMBroadcastTo<T, Context>(
        dev_ctx, y_dims, broadcast_shape_vec, y, &y_broadcast);
  } else {
    experimental::OpCommandHelper::Assign(dev_ctx, y, &y_broadcast);
  }

  experimental::OpCommand("BatchMatMul")
      .Input(x_broadcast,
             experimental::TensorDescMaker("x1", x_broadcast)
                 .SetDataLayout(phi::DataLayout::ANY)
                 .SetDims({batch_size,
                           x_dims[x_dims.size() - 2],
                           x_dims[x_dims.size() - 1]}))
      .Input(y_broadcast,
             experimental::TensorDescMaker("x2", y_broadcast)
                 .SetDataLayout(phi::DataLayout::ANY)
                 .SetDims({batch_size,
                           y_dims[y_dims.size() - 2],
                           y_dims[y_dims.size() - 1]}))
      .Output(*out,
              experimental::TensorDescMaker("y", *out)
                  .SetDataLayout(phi::DataLayout::ANY)
                  .SetDims({batch_size,
                            out_dims[out_dims.size() - 2],
                            out_dims[out_dims.size() - 1]}))
      .Attr("adj_x1", transpose_x)
      .Attr("adj_x2", transpose_y)
      .Run(dev_ctx);
}

template <typename T, typename Context>
static void MMKernel(const Context& dev_ctx,
                     const std::vector<int>& x_dims,
                     const std::vector<int>& y_dims,
                     const std::vector<int>& out_dims,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     bool transpose_x,
                     bool transpose_y,
                     phi::DenseTensor* out) {
  VLOG(10) << "MMKernel: "
           << "(" << phi::make_ddim(x_dims) << "), "
           << "(" << phi::make_ddim(y_dims) << "), "
           << "(" << phi::make_ddim(out_dims) << ")";

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();

  if (x_ndim == 2 && y_ndim == 2) {
    dev_ctx.template Alloc<T>(out);
    experimental::OpCommand("MatMul")
        .Input(x,
               experimental::TensorDescMaker("x1", x)
                   .SetDataLayout(phi::DataLayout::ANY)
                   .SetDims(phi::make_ddim(x_dims)))
        .Input(y,
               experimental::TensorDescMaker("x2", y)
                   .SetDataLayout(phi::DataLayout::ANY)
                   .SetDims(phi::make_ddim(y_dims)))
        .Output(*out,
                experimental::TensorDescMaker("y", *out)
                    .SetDataLayout(phi::DataLayout::ANY)
                    .SetDims(phi::make_ddim(out_dims)))
        .Attr("transpose_x1", transpose_x)
        .Attr("transpose_x2", transpose_y)
        .Run(dev_ctx);
  } else {  // x_ndim == 2 || y_ndim == 2
    if (x_ndim == 2) {
      BMMKernel<T, Context>(dev_ctx,
                            {1, x_dims[0], x_dims[1]},
                            y_dims,
                            out_dims,
                            x,
                            y,
                            transpose_x,
                            transpose_y,
                            out);
    } else {  // y_ndim == 2
      BMMKernel<T, Context>(dev_ctx,
                            x_dims,
                            {1, y_dims[0], y_dims[1]},
                            out_dims,
                            x,
                            y,
                            transpose_x,
                            transpose_y,
                            out);
    }
  }
}

template <typename T, typename Context>
static void MVKernel(const Context& dev_ctx,
                     const std::vector<int>& x_dims,
                     const std::vector<int>& y_dims,
                     const std::vector<int>& out_dims,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     bool transpose_x,
                     bool transpose_y,
                     phi::DenseTensor* out) {
  VLOG(10) << "MVKernel: "
           << "(" << phi::make_ddim(x_dims) << "), "
           << "(" << phi::make_ddim(y_dims) << "), "
           << "(" << phi::make_ddim(out_dims) << ")";

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();

  if (x_ndim == 1 && y_ndim == 1) {
    dev_ctx.template Alloc<T>(out);

    experimental::OpCommand("Dot")
        .Input(x,
               experimental::TensorDescMaker("input_x", x)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Input(y,
               experimental::TensorDescMaker("input_y", y)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Output(*out,
                experimental::TensorDescMaker("output", *out)
                    .SetDataLayout(phi::DataLayout::ANY))
        .Run(dev_ctx);
  } else {  // x_ndim == 1 || y_ndim == 1
    if (x_ndim == 1) {
      std::vector<int> out_dims_temp = out_dims;
      out_dims_temp.insert(out_dims_temp.cend() - 1, 1);
      MMKernel<T, Context>(dev_ctx,
                           {1, x.numel()},
                           y_dims,
                           out_dims_temp,
                           x,
                           y,
                           transpose_x,
                           transpose_y,
                           out);
    } else {
      std::vector<int> out_dims_temp = out_dims;
      out_dims_temp.push_back(1);
      MMKernel<T, Context>(dev_ctx,
                           x_dims,
                           {y.numel(), 1},
                           out_dims_temp,
                           x,
                           y,
                           transpose_x,
                           transpose_y,
                           out);
    }
  }
}

template <typename T, typename Context>
void MatmulKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  phi::DenseTensor* out) {
  auto x_dims = phi::vectorize<int>(x.dims());
  auto y_dims = phi::vectorize<int>(y.dims());
  auto out_dims = phi::vectorize<int>(out->dims());

  if (x_dims.size() == 1 || y_dims.size() == 1) {
    MVKernel<T, Context>(
        dev_ctx, x_dims, y_dims, out_dims, x, y, transpose_x, transpose_y, out);
  } else if (x_dims.size() == 2 || y_dims.size() == 2) {
    MMKernel<T, Context>(
        dev_ctx, x_dims, y_dims, out_dims, x, y, transpose_x, transpose_y, out);
  } else {  // x_dims.size() > 2 && y_dims.size() > 2
    BMMKernel<T, Context>(
        dev_ctx, x_dims, y_dims, out_dims, x, y, transpose_x, transpose_y, out);
  }
}

template <typename T, typename Context>
void BMMGradReduceSum(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const std::vector<int>& reduce_axes,
                      bool keep_dims,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("ReduceSumD")
      .Input(x,
             experimental::TensorDescMaker("x", x).SetDataLayout(
                 phi::DataLayout::ANY))
      .Output(*out,
              experimental::TensorDescMaker("y", *out).SetDataLayout(
                  phi::DataLayout::ANY))
      .Attr("axes", reduce_axes)
      .Attr("keep_dims", keep_dims)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void BMMGradKernel(const Context& dev_ctx,
                   const std::vector<int>& x_dims,
                   const std::vector<int>& y_dims,
                   const std::vector<int>& out_dims,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   const phi::DenseTensor& dout,
                   bool transpose_x,
                   bool transpose_y,
                   phi::DenseTensor* dx,
                   phi::DenseTensor* dy) {
  VLOG(10) << "BMMGradKernel: "
           << "(" << phi::make_ddim(x_dims) << "), "
           << "(" << phi::make_ddim(y_dims) << "), "
           << "(" << phi::make_ddim(out_dims) << ")";

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int out_ndim = out_dims.size();

  auto x_batch_size = std::accumulate(
      x_dims.cbegin(), x_dims.cend() - 2, 1, std::multiplies<int>());
  auto y_batch_size = std::accumulate(
      y_dims.cbegin(), y_dims.cend() - 2, 1, std::multiplies<int>());
  auto batch_size = std::accumulate(
      out_dims.cbegin(), out_dims.cend() - 2, 1, std::multiplies<int>());

  if (dx) {
    if (x_batch_size == batch_size) {
      if (transpose_x) {
        BMMKernel<T, Context>(
            dev_ctx, y_dims, out_dims, x_dims, y, dout, transpose_y, true, dx);
      } else {
        BMMKernel<T, Context>(dev_ctx,
                              out_dims,
                              y_dims,
                              x_dims,
                              dout,
                              y,
                              false,
                              !transpose_y,
                              dx);
      }
    } else {
      std::vector<int> dx_temp_dims(out_dims.cbegin(), out_dims.cend() - 2);
      dx_temp_dims.push_back(x_dims[x_ndim - 2]);
      dx_temp_dims.push_back(x_dims[x_ndim - 1]);

      phi::DenseTensor dx_temp;
      dx_temp.Resize(phi::make_ddim(dx_temp_dims));
      if (transpose_x) {
        BMMKernel<T, Context>(dev_ctx,
                              y_dims,
                              out_dims,
                              dx_temp_dims,
                              y,
                              dout,
                              transpose_y,
                              true,
                              &dx_temp);
      } else {
        BMMKernel<T, Context>(dev_ctx,
                              out_dims,
                              y_dims,
                              dx_temp_dims,
                              dout,
                              y,
                              false,
                              !transpose_y,
                              &dx_temp);
      }

      std::vector<int> reduce_axes;
      for (auto i = 0; i < out_ndim - x_ndim; ++i) {
        reduce_axes.push_back(i);
      }
      phi::DenseTensor dx_temp_reduce_sum;
      if (reduce_axes.size()) {
        dx_temp_reduce_sum.Resize(phi::make_ddim(std::vector<int>(
            dx_temp_dims.cbegin() + out_ndim - x_ndim, dx_temp_dims.cend())));
        BMMGradReduceSum<T, Context>(
            dev_ctx, dx_temp, reduce_axes, false, &dx_temp_reduce_sum);
      } else {
        experimental::OpCommandHelper::Assign(
            dev_ctx, dx_temp, &dx_temp_reduce_sum);
      }

      reduce_axes.clear();
      for (auto i = 0; i < x_ndim - 2; ++i) {
        if (x_dims[i] != out_dims[i + out_ndim - x_ndim]) {
          reduce_axes.push_back(i);
        }
      }
      if (reduce_axes.size()) {
        BMMGradReduceSum<T, Context>(
            dev_ctx, dx_temp_reduce_sum, reduce_axes, true, dx);
      } else {
        dev_ctx.template Alloc<T>(dx);
        experimental::OpCommandHelper::Assign(dev_ctx, dx_temp_reduce_sum, dx);
      }
    }
  }
  if (dy) {
    if (y_batch_size == batch_size) {
      if (transpose_y) {
        BMMKernel<T, Context>(
            dev_ctx, out_dims, x_dims, y_dims, dout, x, true, transpose_x, dy);
      } else {
        BMMKernel<T, Context>(dev_ctx,
                              x_dims,
                              out_dims,
                              y_dims,
                              x,
                              dout,
                              !transpose_x,
                              false,
                              dy);
      }
    } else {
      std::vector<int> dy_temp_dims(out_dims.cbegin(), out_dims.cend() - 2);
      dy_temp_dims.push_back(y_dims[y_ndim - 2]);
      dy_temp_dims.push_back(y_dims[y_ndim - 1]);

      phi::DenseTensor dy_temp;
      dy_temp.Resize(phi::make_ddim(dy_temp_dims));

      if (transpose_y) {
        BMMKernel<T, Context>(dev_ctx,
                              out_dims,
                              x_dims,
                              dy_temp_dims,
                              dout,
                              x,
                              true,
                              transpose_x,
                              &dy_temp);
      } else {
        BMMKernel<T, Context>(dev_ctx,
                              x_dims,
                              out_dims,
                              dy_temp_dims,
                              x,
                              dout,
                              !transpose_x,
                              false,
                              &dy_temp);
      }

      std::vector<int> reduce_axes;
      for (auto i = 0; i < out_ndim - y_ndim; ++i) {
        reduce_axes.push_back(i);
      }
      phi::DenseTensor dy_temp_reduce_sum;
      if (reduce_axes.size()) {
        dy_temp_reduce_sum.Resize(phi::make_ddim(std::vector<int>(
            dy_temp_dims.cbegin() + out_ndim - y_ndim, dy_temp_dims.cend())));
        BMMGradReduceSum<T, Context>(
            dev_ctx, dy_temp, reduce_axes, false, &dy_temp_reduce_sum);
      } else {
        experimental::OpCommandHelper::Assign(
            dev_ctx, dy_temp, &dy_temp_reduce_sum);
      }

      reduce_axes.clear();
      for (auto i = 0; i < y_ndim - 2; ++i) {
        if (y_dims[i] != out_dims[i + out_ndim - y_ndim]) {
          reduce_axes.push_back(i);
        }
      }
      if (reduce_axes.size()) {
        BMMGradReduceSum<T, Context>(
            dev_ctx, dy_temp_reduce_sum, reduce_axes, true, dy);
      } else {
        dev_ctx.template Alloc<T>(dy);
        experimental::OpCommandHelper::Assign(dev_ctx, dy_temp_reduce_sum, dy);
      }
    }
  }
}

template <typename T, typename Context>
void MMGradKernel(const Context& dev_ctx,
                  const std::vector<int>& x_dims,
                  const std::vector<int>& y_dims,
                  const std::vector<int>& out_dims,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  const phi::DenseTensor& dout,
                  bool transpose_x,
                  bool transpose_y,
                  phi::DenseTensor* dx,
                  phi::DenseTensor* dy) {
  VLOG(10) << "MMGradKernel: "
           << "(" << phi::make_ddim(x_dims) << "), "
           << "(" << phi::make_ddim(y_dims) << "), "
           << "(" << phi::make_ddim(out_dims) << ")";

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int out_ndim = out_dims.size();

  if (out_ndim == 2) {
    if (dx) {
      if (transpose_x) {
        MMKernel<T, Context>(
            dev_ctx, y_dims, out_dims, x_dims, y, dout, transpose_y, true, dx);
      } else {
        MMKernel<T, Context>(dev_ctx,
                             out_dims,
                             y_dims,
                             x_dims,
                             dout,
                             y,
                             false,
                             !transpose_y,
                             dx);
      }
    }
    if (dy) {
      if (transpose_y) {
        MMKernel<T, Context>(
            dev_ctx, out_dims, x_dims, y_dims, dout, x, true, transpose_x, dy);
      } else {
        MMKernel<T, Context>(dev_ctx,
                             x_dims,
                             out_dims,
                             y_dims,
                             x,
                             dout,
                             !transpose_x,
                             false,
                             dy);
      }
    }
  } else {  // x_ndim == 2 || y_ndim == 2
    if (x_ndim == 2) {
      auto x_dims_tmp = x_dims;
      x_dims_tmp.insert(x_dims_tmp.cbegin(), 1);
      BMMGradKernel<T, Context>(dev_ctx,
                                x_dims_tmp,
                                y_dims,
                                out_dims,
                                x,
                                y,
                                dout,
                                transpose_x,
                                transpose_y,
                                dx,
                                dy);
    } else {  // y_ndim == 2
      auto y_dims_tmp = y_dims;
      y_dims_tmp.insert(y_dims_tmp.cbegin(), 1);
      BMMGradKernel<T, Context>(dev_ctx,
                                x_dims,
                                y_dims_tmp,
                                out_dims,
                                x,
                                y,
                                dout,
                                transpose_x,
                                transpose_y,
                                dx,
                                dy);
    }
  }
}

template <typename T, typename Context>
void MVGradKernel(const Context& dev_ctx,
                  const std::vector<int>& x_dims,
                  const std::vector<int>& y_dims,
                  const std::vector<int>& out_dims,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  const phi::DenseTensor& dout,
                  bool transpose_x,
                  bool transpose_y,
                  phi::DenseTensor* dx,
                  phi::DenseTensor* dy) {
  VLOG(10) << "MVGradKernel: "
           << "(" << phi::make_ddim(x_dims) << "), "
           << "(" << phi::make_ddim(y_dims) << "), "
           << "(" << phi::make_ddim(out_dims) << ")";

  int x_ndim = x_dims.size();
  int out_ndim = out_dims.size();

  if (out_ndim == 1) {
    if (dx) {
      dev_ctx.template Alloc<T>(dx);
      experimental::OpCommand("Mul")
          .Input(dout,
                 experimental::TensorDescMaker("x1", dout)
                     .SetDataLayout(phi::DataLayout::ANY))
          .Input(y,
                 experimental::TensorDescMaker("x2", y).SetDataLayout(
                     phi::DataLayout::ANY))
          .Output(*dx,
                  experimental::TensorDescMaker("y", *dx).SetDataLayout(
                      phi::DataLayout::ANY))
          .Run(dev_ctx);
    }
    if (dy) {
      dev_ctx.template Alloc<T>(dy);
      experimental::OpCommand("Mul")
          .Input(dout,
                 experimental::TensorDescMaker("x1", dout)
                     .SetDataLayout(phi::DataLayout::ANY))
          .Input(x,
                 experimental::TensorDescMaker("x2", x).SetDataLayout(
                     phi::DataLayout::ANY))
          .Output(*dy,
                  experimental::TensorDescMaker("y", *dy).SetDataLayout(
                      phi::DataLayout::ANY))
          .Run(dev_ctx);
    }
  } else {  // x_ndim == 1 || y_ndim == 1
    if (x_ndim == 1) {
      auto out_dims_tmp = out_dims;
      out_dims_tmp.insert(out_dims_tmp.cend() - 1, 1);
      MMGradKernel<T, Context>(dev_ctx,
                               {1, x.numel()},
                               y_dims,
                               out_dims_tmp,
                               x,
                               y,
                               dout,
                               transpose_x,
                               transpose_y,
                               dx,
                               dy);
    } else {
      auto out_dims_tmp = out_dims;
      out_dims_tmp.push_back(1);
      MMGradKernel<T, Context>(dev_ctx,
                               x_dims,
                               {y.numel(), 1},
                               out_dims_tmp,
                               x,
                               y,
                               dout,
                               transpose_x,
                               transpose_y,
                               dx,
                               dy);
    }
  }
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
  auto x_dims = phi::vectorize<int>(x.dims());
  auto y_dims = phi::vectorize<int>(y.dims());
  auto out_dims = phi::vectorize<int>(dout.dims());

  if (x_dims.size() == 1 || y_dims.size() == 1) {
    MVGradKernel<T, Context>(dev_ctx,
                             x_dims,
                             y_dims,
                             out_dims,
                             x,
                             y,
                             dout,
                             transpose_x,
                             transpose_y,
                             dx,
                             dy);
  } else if (x_dims.size() == 2 || y_dims.size() == 2) {
    MMGradKernel<T, Context>(dev_ctx,
                             x_dims,
                             y_dims,
                             out_dims,
                             x,
                             y,
                             dout,
                             transpose_x,
                             transpose_y,
                             dx,
                             dy);
  } else {  // x_dims.size() > 2 && y_dims.size() > 2
    BMMGradKernel<T, Context>(dev_ctx,
                              x_dims,
                              y_dims,
                              out_dims,
                              x,
                              y,
                              dout,
                              transpose_x,
                              transpose_y,
                              dx,
                              dy);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(matmul,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MatmulKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(matmul_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MatmulGradKernel,
                          float,
                          phi::dtype::float16) {}
