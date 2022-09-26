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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T,typename Context>
static void Mul(const Context& dev_ctx,
                const phi::DenseTensor& X,
                const phi::DenseTensor& Y,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  float alpha = 1.0;

  MLUCnnlTensorDesc x_desc(X, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(Y, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  MLUCnnlOpTensorDesc mul_op_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
  MLUCnnl::OpTensor(dev_ctx,
                    mul_op_desc.get(),
                    x_desc.get(),
                    GetBasePtr(&X),
                    y_desc.get(),
                    GetBasePtr(&Y),
                    out_desc.get(),
                    GetBasePtr(out),
                    ToCnnlDataType<T>(),
                    alpha);
}

template <typename T,typename Context>
static void MatMul2D(const Context& dev_ctx,
                     const phi::DenseTensor& X,
                     const phi::DenseTensor& Y,
                     phi::DenseTensor* out,
                     const bool transpose_x,
                     const bool transpose_y) {
  dev_ctx.template Alloc<T>(out);
  MLUCnnlTensorDesc x_desc(X, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(Y, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  MLUCnnl::Matmul(dev_ctx,
                  transpose_x,
                  transpose_y,
                  x_desc.get(),
                  GetBasePtr(&X),
                  y_desc.get(),
                  GetBasePtr(&Y),
                  out_desc.get(),
                  GetBasePtr(out));
}


template <typename T, typename Context>
static void MatMulND(const Context& dev_ctx,
                     const phi::DenseTensor& X,
                     const phi::DenseTensor& Y,
                     phi::DenseTensor* out,
                     const bool transpose_x,
                     const bool transpose_y) {
  if (!out->initialized()) {
    dev_ctx.template Alloc<T>(out);
  }
  MLUCnnlTensorDesc x_desc(X, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(Y, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  MLUCnnl::BatchMatmul(dev_ctx,
                       transpose_x,
                       transpose_y,
                       x_desc.get(),
                       GetBasePtr(&X),
                       y_desc.get(),
                       GetBasePtr(&Y),
                       out_desc.get(),
                       GetBasePtr(out));
}

template <typename T, typename Context>
static void ReduceDims(const Context& dev_ctx,
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

  MLUCnnlTensorDesc in_desc(in, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  std::vector<int> reduce_dims(axes.begin(), axes.end());
  MLUCnnlReduceDesc reduce_desc(reduce_dims,
                                CNNL_REDUCE_ADD,
                                ToCnnlDataType<T>(),
                                CNNL_NOT_PROPAGATE_NAN,
                                CNNL_REDUCE_NO_INDICES,
                                CNNL_32BIT_INDICES);

  MLUCnnl::Reduce(dev_ctx,
                  true /*need_workspace*/,
                  reduce_desc.get(),
                  nullptr,
                  in_desc.get(),
                  GetBasePtr(&in),
                  0 /*indices_size*/,
                  nullptr,
                  nullptr,
                  out_desc.get(),
                  GetBasePtr(out));
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
    // Case 1: [K] x [K] = [1]
    // Equal: [1, K] x [K, 1] = [1, 1] => [1]
    const bool all_one_dim = (x_ndim == 1 && y_ndim == 1);
    if (all_one_dim) {
      out->Resize({1, 1});
    }

    // Resize dim 1 to 2
    Tensor x_temp, y_temp;
    x_temp = x;
    y_temp = y;
    if (x_ndim == 1) {
      x_dims.insert(x_dims.begin(), 1);
      x_temp.Resize(phi::make_ddim(x_dims));
      x_ndim = 2;
      // matmul op of mlu needs `std::max(x->dim, y->dim) == out->dim`
      if (out_dims.size() < y_dims.size()) {
        std::vector<int64_t> temp_out_dims(out_dims.begin(), out_dims.end());
        temp_out_dims.insert(temp_out_dims.end() - 1, 1);
        out->Resize(phi::make_ddim(temp_out_dims));
      }
    }
    if (y_ndim == 1) {
      y_dims.push_back(1);
      y_temp.Resize(phi::make_ddim(y_dims));
      y_ndim = 2;
      // matmul op of mlu needs `std::max(x->dim, y->dim) == out->dim`
      if (out_dims.size() < x_dims.size()) {
        std::vector<int64_t> temp_out_dims(out_dims.begin(), out_dims.end());
        temp_out_dims.push_back(1);
        out->Resize(phi::make_ddim(temp_out_dims));
      }
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

    if (x_ndim == 2 && y_ndim == 2) {
      // Case 2: [M, K] x [K, N] = [M, N]
      MatMul2D<T>(dev_ctx, x_temp, y_temp, out, transpose_x, transpose_y);
    } else {
      // Case 3: [B, M, K] x [K, N] =  [B, M, N]
      // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
      MatMulND<T>(dev_ctx, x_temp, y_temp, out, transpose_x, transpose_y);
    }

    if (phi::vectorize(out->dims()) != out_dims) {
      out->Resize(phi::make_ddim(out_dims));
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
    std::vector<int64_t> x_dims = phi::vectorize(x.dims());
    std::vector<int64_t> y_dims = phi::vectorize(y.dims());
    std::vector<int64_t> out_dims = phi::vectorize(dout.dims());
    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();
    int out_ndim = out_dims.size();

    // Case 1: [K] x [K] = [1]
    if (x_ndim == 1 && y_ndim == 1) {
      if (dx) {
        Mul<T>(dev_ctx, dout, y, dx);
      }
      if (dy) {
        Mul<T>(dev_ctx, dout, x, dy);
      }
      return;
    }

    // Resize dim 1 to 2
    Tensor x_temp, y_temp, dout_temp;
    x_temp = x;
    y_temp = y;
    dout_temp = dout;
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
          MatMul2D<T>(dev_ctx, y_temp, dout_temp, dx, transpose_y, true);
        } else {
          MatMul2D<T>(dev_ctx, dout_temp, y_temp, dx, false, !transpose_y);
        }
        dx->Resize(x.dims());
      }
      if (dy) {
        dy->Resize(phi::make_ddim(y_dims));
        if (transpose_y) {
          MatMul2D<T>(dev_ctx, dout_temp, x_temp, dy, true, transpose_x);
        } else {
          MatMul2D<T>(dev_ctx, x_temp, dout_temp, dy, !transpose_x, false);
        }
        dy->Resize(y.dims());
      }
      return;
    }

    // Case 3: [B, M, K] x [K, N] =  [B, M, N]
    // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
    std::vector<int64_t> x_bcast_dims(out_ndim, 1);
    std::vector<int64_t> y_bcast_dims(out_ndim, 1);
    std::copy(out_dims.begin(), out_dims.end() - 2, x_bcast_dims.begin());
    std::copy(out_dims.begin(), out_dims.end() - 2, y_bcast_dims.begin());
    std::copy(x_dims.end() - 2, x_dims.end(), x_bcast_dims.end() - 2);
    std::copy(y_dims.end() - 2, y_dims.end(), y_bcast_dims.end() - 2);

    if (dx) {
      Tensor dx_temp;
      if (x_dims != x_bcast_dims) {
        dx_temp.Resize(phi::make_ddim(x_bcast_dims));
      } else {
        dev_ctx.template Alloc<T>(dx);
        dx_temp = *dx;
      }

      if (transpose_x) {
        MatMulND<T>(dev_ctx, y_temp, dout_temp, &dx_temp, transpose_y, true);
      } else {
        MatMulND<T>(
            dev_ctx, dout_temp, y_temp, &dx_temp, false, !transpose_y);
      }

      if (x_dims != x_bcast_dims) {
        ReduceDims<T>(dev_ctx, x_dims, x_bcast_dims, dx_temp, dx);
      }
    }

    if (dy) {
      Tensor dy_temp;
      if (y_dims != y_bcast_dims) {
        dy_temp.Resize(phi::make_ddim(y_bcast_dims));
      } else {
        dev_ctx.template Alloc<T>(dy);
        dy_temp = *dy;
      }

      if (transpose_y) {
        MatMulND<T>(dev_ctx, dout_temp, x_temp, &dy_temp, true, transpose_x);
      } else {
        MatMulND<T>(
            dev_ctx, x_temp, dout_temp, &dy_temp, !transpose_x, false);
      }

      if (y_dims != y_bcast_dims) {
        ReduceDims<T>(dev_ctx, y_dims, y_bcast_dims, dy_temp, dy);
      }
    }

}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(matmul,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::MatmulKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(matmul_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::MatmulGradKernel,
                          float,
                          phi::dtype::float16) {}
