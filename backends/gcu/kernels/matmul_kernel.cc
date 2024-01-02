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

#include "kernels/common_ops/common_ops.h"
#include "kernels/common_ops/elementwise_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void MatmulKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  bool trans_x,
                  bool trans_y,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "matmul", matmul);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    auto x_shape = phi::vectorize(x.dims());
    auto y_shape = phi::vectorize(y.dims());
    int64_t x_rank = x_shape.size();
    int64_t y_rank = y_shape.size();
    int64_t max_rank = std::max(x_rank, y_rank);
    int64_t rank_diff = std::abs(x_rank - y_rank);
    int64_t batch_dim;

    if (x_rank > y_rank) {
      if (trans_x || y_rank == 1) {
        std::vector<int64_t> broadcast_dims;
        std::vector<int64_t> bc_shape;
        if (y_rank == 1) {
          for (int64_t i = 0; i < rank_diff - 1; i++) {
            bc_shape.emplace_back(x_shape[i]);
          }
          bc_shape.emplace_back(y_shape[0]);
          bc_shape.emplace_back(1);
          broadcast_dims.emplace_back(rank_diff - 1);
        } else {
          for (int64_t i = 0; i < rank_diff; i++) {
            bc_shape.emplace_back(x_shape[i]);
          }
          for (int64_t i = 0; i < y_rank; i++) {
            bc_shape.emplace_back(y_shape[i]);
          }
          int iter = 0;
          for (int64_t i = 0; i < x_rank; ++i) {
            if (i < rank_diff) {
              ++iter;
            } else {
              broadcast_dims.emplace_back(i);
            }
          }
        }
        tmp_y = broadcast_in_dim(dev_ctx, tmp_y, bc_shape, broadcast_dims);
      }
      if (y_rank == 1) {
        batch_dim = rank_diff - 1;
      } else {
        batch_dim = rank_diff;
      }

    } else if (x_rank < y_rank) {
      std::vector<int64_t> broadcast_dims;
      std::vector<int64_t> bc_shape;
      if (x_rank == 1) {
        for (int64_t i = 0; i < rank_diff - 1; i++) {
          bc_shape.emplace_back(y_shape[i]);
        }
        bc_shape.emplace_back(1);
        bc_shape.emplace_back(x_shape[0]);
        broadcast_dims.emplace_back(rank_diff);
      } else {
        for (int64_t i = 0; i < rank_diff; i++) {
          bc_shape.emplace_back(y_shape[i]);
        }
        for (int64_t i = 0; i < x_rank; i++) {
          bc_shape.emplace_back(x_shape[i]);
        }
        int iter = 0;
        for (int64_t i = 0; i < y_rank; ++i) {
          if (i < rank_diff) {
            ++iter;
          } else {
            broadcast_dims.emplace_back(i);
          }
        }
      }
      tmp_x = broadcast_in_dim(dev_ctx, tmp_x, bc_shape, broadcast_dims);
      if (x_rank == 1) {
        batch_dim = rank_diff - 1;
      } else {
        batch_dim = rank_diff;
      }

    } else {
      batch_dim = max_rank - 2;
      if (x_rank == y_rank && x_rank > 3) {
        auto x_brd_shape = x_shape;
        auto y_brd_shape = y_shape;
        std::vector<int64_t> x_brd_dims, y_brd_dims;
        for (int64_t i = 0; i < x_rank - 2; ++i) {
          x_brd_shape[i] = x_shape[i] > y_shape[i] ? x_shape[i] : y_shape[i];
          y_brd_shape[i] = x_shape[i] > y_shape[i] ? x_shape[i] : y_shape[i];
        }
        x_brd_dims.resize(x_rank);
        y_brd_dims.resize(y_rank);
        std::iota(x_brd_dims.begin(), x_brd_dims.end(), 0);
        std::iota(y_brd_dims.begin(), y_brd_dims.end(), 0);
        if (x_brd_shape != x_shape) {
          tmp_x = broadcast_in_dim(dev_ctx, tmp_x, x_brd_shape, x_brd_dims);
        }
        if (y_brd_shape != y_shape) {
          tmp_y = broadcast_in_dim(dev_ctx, tmp_y, y_brd_shape, y_brd_dims);
        }
      }
    }

    builder::DotDimensionNumbers dims_attr;
    std::vector<int64_t> lhs_batching_dimensions = {};
    std::vector<int64_t> rhs_batching_dimensions = {};
    std::vector<int64_t> lhs_contracting_dimensions = {};
    std::vector<int64_t> rhs_contracting_dimensions = {};
    if (x_rank == 1 && y_rank == 1) {
      lhs_contracting_dimensions.emplace_back(0);
      rhs_contracting_dimensions.emplace_back(0);
    } else if (x_rank <= y_rank || trans_x || y_rank == 1) {
      for (int64_t i = 0; i < max_rank - 1; ++i) {
        if (i < batch_dim) {
          lhs_batching_dimensions.emplace_back(i);
          rhs_batching_dimensions.emplace_back(i);
        } else {
          if (trans_x && x_rank != 1) {
            lhs_contracting_dimensions.emplace_back(i);
          } else {
            lhs_contracting_dimensions.emplace_back(i + 1);
          }
          if (trans_y && y_rank != 1) {
            rhs_contracting_dimensions.emplace_back(i + 1);
          } else {
            rhs_contracting_dimensions.emplace_back(i);
          }
        }
      }
    } else {
      lhs_contracting_dimensions.emplace_back(x_rank - 1);
      if (y_rank != 1) {
        if (trans_y) {
          rhs_contracting_dimensions.emplace_back(y_rank - 1);
        } else {
          rhs_contracting_dimensions.emplace_back(y_rank - 2);
        }
      } else {
        rhs_contracting_dimensions.emplace_back(0);
      }
    }

    *out = dot_general_common(dev_ctx,
                              tmp_x,
                              tmp_y,
                              lhs_batching_dimensions,
                              rhs_batching_dimensions,
                              lhs_contracting_dimensions,
                              rhs_contracting_dimensions);

    if (x_rank == 1 && y_rank == 1) {
      std::vector<int64_t> new_shape;
      new_shape.push_back(1);
      *out = reshape(dev_ctx, *out, new_shape);
    } else if (y_rank == 1) {
      auto shape = phi::vectorize(out->dims());
      std::vector<int64_t> new_shape;
      for (size_t i = 0; i < shape.size() - 1; i++) {
        new_shape.push_back(shape[i]);
      }
      *out = reshape(dev_ctx, *out, new_shape);
    } else if (x_rank == 1) {
      auto shape = phi::vectorize(out->dims());
      std::vector<int64_t> new_shape;
      for (size_t i = 0; i < shape.size(); i++) {
        if (i != shape.size() - 2) {
          new_shape.push_back(shape[i]);
        }
      }
      *out = reshape(dev_ctx, *out, new_shape);
    }
    PADDLE_GCU_KERNEL_END("matmul", matmul);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Y"] = {"y"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Y"] = {const_cast<DenseTensor*>(&y)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["trans_x"] = trans_x;
    attrs["trans_y"] = trans_y;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "matmul_v2",
              dev_ctx);
  }
}

template <typename T, typename Context>
void MatmulGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      const phi::DenseTensor& dout,
                      bool trans_x,
                      bool trans_y,
                      phi::DenseTensor* dx,
                      phi::DenseTensor* dy) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "matmul_grad", matmul_grad);

    auto x_shape = phi::vectorize(x.dims());
    auto y_shape = phi::vectorize(y.dims());
    auto out_shape = phi::vectorize(dout.dims());
    int64_t x_rank = x_shape.size();
    int64_t y_rank = y_shape.size();
    int64_t out_rank = out_shape.size();
    int64_t max_rank = std::max(x_rank, y_rank);
    int64_t rank_diff = std::abs(x_rank - y_rank);
    int64_t batch_dim;

    // broadcast X, Y
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    phi::DenseTensor tmp_dout = dout;

    if (x_rank > y_rank) {
      if (trans_x || y_rank == 1) {
        std::vector<int64_t> broadcast_dims;
        std::vector<int64_t> bc_shape;
        if (y_rank == 1) {
          for (int64_t i = 0; i < rank_diff - 1; i++) {
            bc_shape.emplace_back(x_shape[i]);
          }
          bc_shape.emplace_back(y_shape[0]);
          bc_shape.emplace_back(1);
          broadcast_dims.emplace_back(rank_diff - 1);
        } else {
          for (int64_t i = 0; i < rank_diff; i++) {
            bc_shape.emplace_back(x_shape[i]);
          }
          for (int64_t i = 0; i < y_rank; i++) {
            bc_shape.emplace_back(y_shape[i]);
          }
          int iter = 0;
          for (int64_t i = 0; i < x_rank; ++i) {
            if (i < rank_diff) {
              ++iter;
            } else {
              broadcast_dims.emplace_back(i);
            }
          }
        }
        tmp_y = broadcast_in_dim(dev_ctx, tmp_y, bc_shape, broadcast_dims);
      }
      if (y_rank == 1) {
        batch_dim = rank_diff - 1;
      } else {
        batch_dim = rank_diff;
      }
    } else if (x_rank < y_rank) {
      std::vector<int64_t> broadcast_dims;
      std::vector<int64_t> bc_shape;
      if (x_rank == 1) {
        for (int64_t i = 0; i < rank_diff - 1; i++) {
          bc_shape.emplace_back(y_shape[i]);
        }
        bc_shape.emplace_back(1);
        bc_shape.emplace_back(x_shape[0]);
        broadcast_dims.emplace_back(rank_diff);
      } else {
        for (int64_t i = 0; i < rank_diff; i++) {
          bc_shape.emplace_back(y_shape[i]);
        }
        for (int64_t i = 0; i < x_rank; i++) {
          bc_shape.emplace_back(x_shape[i]);
        }
        int iter = 0;
        for (int64_t i = 0; i < y_rank; ++i) {
          if (i < rank_diff) {
            ++iter;
          } else {
            broadcast_dims.emplace_back(i);
          }
        }
      }
      tmp_x = broadcast_in_dim(dev_ctx, tmp_x, bc_shape, broadcast_dims);
      if (x_rank == 1) {
        batch_dim = rank_diff - 1;
      } else {
        batch_dim = rank_diff;
      }
    } else {
      batch_dim = max_rank - 2;
      if (x_rank == y_rank && x_rank > 3) {
        auto x_brd_shape = x_shape;
        auto y_brd_shape = y_shape;
        std::vector<int64_t> x_brd_dims, y_brd_dims;
        for (int64_t i = 0; i < x_rank - 2; ++i) {
          x_brd_shape[i] = x_shape[i] > y_shape[i] ? x_shape[i] : y_shape[i];
          y_brd_shape[i] = x_shape[i] > y_shape[i] ? x_shape[i] : y_shape[i];
        }
        x_brd_dims.resize(x_rank);
        y_brd_dims.resize(y_rank);
        std::iota(x_brd_dims.begin(), x_brd_dims.end(), 0);
        std::iota(y_brd_dims.begin(), y_brd_dims.end(), 0);
        if (x_brd_shape != x_shape) {
          tmp_x = broadcast_in_dim(dev_ctx, tmp_x, x_brd_shape, x_brd_dims);
        }
        if (y_brd_shape != y_shape) {
          tmp_y = broadcast_in_dim(dev_ctx, tmp_y, y_brd_shape, y_brd_dims);
        }
      }
    }

    // reshape out@grad
    if (y_rank == 1 && x_rank == 1) {
      std::vector<int64_t> new_shape;
      new_shape.emplace_back(1);
      tmp_dout = reshape(dev_ctx, tmp_dout, new_shape);
    } else if (y_rank == 1) {
      std::vector<int64_t> new_shape;
      for (size_t i = 0; i < out_shape.size(); i++) {
        new_shape.emplace_back(out_shape[i]);
      }
      new_shape.emplace_back(1);
      tmp_dout = reshape(dev_ctx, tmp_dout, new_shape);
    } else if (x_rank == 1) {
      std::vector<int64_t> new_shape;
      for (size_t i = 0; i < out_shape.size() - 1; i++) {
        new_shape.emplace_back(out_shape[i]);
      }
      new_shape.emplace_back(1);
      new_shape.emplace_back(out_shape[out_shape.size() - 1]);
      tmp_dout = reshape(dev_ctx, tmp_dout, new_shape);
    }

    // calculate DX, DY
    if (y_rank == 1 && x_rank == 1) {
      *dx = mul_compute(dev_ctx, tmp_dout, tmp_y);
      *dy = mul_compute(dev_ctx, tmp_x, tmp_dout);
    } else {
      std::vector<int64_t> lhs_batching_dimensions_dx = {};
      std::vector<int64_t> rhs_batching_dimensions_dx = {};
      std::vector<int64_t> lhs_contracting_dimensions_dx = {};
      std::vector<int64_t> rhs_contracting_dimensions_dx = {};
      if (out_rank <= y_rank || trans_x || y_rank == 1) {
        for (int64_t i = 0; i < max_rank - 1; ++i) {
          if (i < batch_dim) {
            lhs_batching_dimensions_dx.emplace_back(i);
            rhs_batching_dimensions_dx.emplace_back(i);
          } else {
            lhs_contracting_dimensions_dx.emplace_back(i + 1);
            if (trans_y && y_rank != 1) {
              rhs_contracting_dimensions_dx.emplace_back(i);
            } else {
              rhs_contracting_dimensions_dx.emplace_back(i + 1);
            }
          }
        }
      } else {
        lhs_contracting_dimensions_dx.emplace_back(out_rank - 1);
        if (y_rank != 1) {
          if (trans_y) {
            rhs_contracting_dimensions_dx.emplace_back(y_rank - 2);
          } else {
            rhs_contracting_dimensions_dx.emplace_back(y_rank - 1);
          }
        } else {
          rhs_contracting_dimensions_dx.emplace_back(0);
        }
      }

      if (dx) {
        *dx = dot_general_common(dev_ctx,
                                 tmp_dout,
                                 tmp_y,
                                 lhs_batching_dimensions_dx,
                                 rhs_batching_dimensions_dx,
                                 lhs_contracting_dimensions_dx,
                                 rhs_contracting_dimensions_dx);
      }

      if (x_rank == y_rank && x_rank > 3) {
        auto dx_shape = phi::vectorize(dx->dims());
        auto true_dx_shape = x_shape;
        if (trans_x) {
          true_dx_shape[x_rank - 2] = x_shape[x_rank - 1];
          true_dx_shape[x_rank - 1] = x_shape[x_rank - 2];
        }
        if (dx_shape != true_dx_shape) {
          std::vector<int64_t> axis;
          for (int64_t i = 0; i < x_rank; ++i) {
            if (dx_shape[i] != true_dx_shape[i]) axis.push_back(i);
          }
          *dx = reduce_sum_compute(dev_ctx, *dx, true, axis);
        }
      }

      std::vector<int64_t> lhs_batching_dimensions_dy = {};
      std::vector<int64_t> rhs_batching_dimensions_dy = {};
      std::vector<int64_t> lhs_contracting_dimensions_dy = {};
      std::vector<int64_t> rhs_contracting_dimensions_dy = {};
      for (int64_t i = 0; i < max_rank - 1; ++i) {
        if (i < batch_dim) {
          lhs_batching_dimensions_dy.emplace_back(i);
          rhs_batching_dimensions_dy.emplace_back(i);
        } else {
          if (trans_x && x_rank != 1) {
            lhs_contracting_dimensions_dy.emplace_back(i + 1);
          } else {
            lhs_contracting_dimensions_dy.emplace_back(i);
          }
          rhs_contracting_dimensions_dy.emplace_back(i);
        }
      }
      if (dy) {
        *dy = dot_general_common(dev_ctx,
                                 tmp_x,
                                 tmp_dout,
                                 lhs_batching_dimensions_dy,
                                 rhs_batching_dimensions_dy,
                                 lhs_contracting_dimensions_dy,
                                 rhs_contracting_dimensions_dy);
      }

      if (x_rank == y_rank && x_rank > 3) {
        auto dy_shape = phi::vectorize(dy->dims());
        auto true_dy_shape = y_shape;
        if (trans_y) {
          true_dy_shape[y_rank - 2] = y_shape[y_rank - 1];
          true_dy_shape[y_rank - 1] = y_shape[y_rank - 2];
        }
        if (dy_shape != true_dy_shape) {
          std::vector<int64_t> axis;
          for (int64_t i = 0; i < x_rank; ++i) {
            if (dy_shape[i] != true_dy_shape[i]) axis.push_back(i);
          }
          *dy = reduce_sum_compute(dev_ctx, *dy, true, axis);
        }
      }
    }

    // transpose back to original input shape(trans_x or trans_y)
    std::vector<int64_t> data_trans;
    for (int64_t i = 0; i < max_rank - 2; ++i) {
      data_trans.emplace_back(i);
    }
    data_trans.emplace_back(max_rank - 1);
    data_trans.emplace_back(max_rank - 2);
    if (trans_x && x_rank != 1) {
      *dx = transpose(dev_ctx, *dx, data_trans);
    }
    if (trans_y && y_rank != 1) {
      *dy = transpose(dev_ctx, *dy, data_trans);
    }
    // reduce sum when x_rank != y_rank
    if (x_rank > y_rank) {
      std::vector<int64_t> axis;
      if (batch_dim != 0) {
        for (int64_t i = 0; i < batch_dim; ++i) {
          axis.emplace_back(i);
        }
        *dy = reduce_sum_compute(dev_ctx, *dy, false, axis);
      }
    } else if (x_rank < y_rank) {
      std::vector<int64_t> axis;
      if (batch_dim != 0) {
        for (int64_t i = 0; i < batch_dim; ++i) {
          axis.emplace_back(i);
        }
        *dx = reduce_sum_compute(dev_ctx, *dx, false, axis);
      }
    }

    // reshape when (x_rank ==1 or y_rank==1) and (x_rank != y_rank)
    if (x_rank == 1 && y_rank != 1) {
      auto dx_shape = phi::vectorize(dx->dims());
      std::vector<int64_t> dx_new_shape;
      dx_new_shape.push_back(dx_shape[dx_shape.size() - 1]);
      *dx = reshape(dev_ctx, *dx, dx_new_shape);
    } else if (y_rank == 1 && x_rank != 1) {
      auto dy_shape = phi::vectorize(dy->dims());
      std::vector<int64_t> dy_new_shape;
      dy_new_shape.push_back(dy_shape[0]);
      *dy = reshape(dev_ctx, *dy, dy_new_shape);
    }
    PADDLE_GCU_KERNEL_END("matmul_grad", matmul_grad);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Y"] = {"y"};
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Y"] = {const_cast<DenseTensor*>(&y)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    TensorValueMap outputs;
    if (dx) {
      dev_ctx.template Alloc<T>(dx);
      output_names[GradVarName("X")] = {"dx"};
      outputs[GradVarName("X")] = {dx};
    }
    if (dy) {
      dev_ctx.template Alloc<T>(dy);
      output_names[GradVarName("Y")] = {"dy"};
      outputs[GradVarName("Y")] = {dy};
    }

    GcuAttributeMap attrs;
    attrs["trans_x"] = trans_x;
    attrs["trans_y"] = trans_y;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "matmul_v2_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(matmul,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MatmulKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(matmul_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MatmulGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
