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

#include "graph/graph_executor.h"

namespace custom_graph {

class MatmulV2Adapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& X = ctx.Input("X");
    auto& Y = ctx.Input("Y");
    auto& Out = ctx.Output("Out");
    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    auto x_dims = X.Shape();
    auto y_dims = Y.Shape();
    auto out_dims = Out.Shape();
    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();
    int out_ndim = out_dims.size();

    if (x_ndim == 1 && y_ndim == 1) {
      //   PADDLE_ENFORCE_EQ(
      //       X.Numel(),
      //       Y.Numel(),
      //       platform::errors::InvalidArgument(
      //           "X's numbers must be equal to Y's numbers,"
      //           "when X/Y's dims =1. But received X has [%d] elements,"
      //           "received Y has [%d] elements",
      //           X.Numel(),
      //           Y.Numel()));

      OpCommand("Dot")
          .Input(X, "input_x")
          .Input(Y, "input_y")
          .Output(Out, "output");
      return;
    }

    // Case 2: [M, K] x [K, N] = [M, N]
    if (x_ndim == 2 && y_ndim == 2) {
      OpCommand("MatMul")
          .Input(X, "x1")
          .Input(Y, "x2")
          .Output(Out, "y")
          .Attr("transpose_x1", trans_x)
          .Attr("transpose_x2", trans_y);
      return;
    }

    // Case 3: [B, M, K] x [K, N] =  [B, M, N], when trans_x = false
    // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
    if (trans_x == false && y_ndim == 2) {
      Tensor tmp_x, tmp_out;
      OpCommand::Reshape(X,
                         tmp_x,
                         {static_cast<int32_t>(x_dims[0] * x_dims[1]),
                          static_cast<int32_t>(x_dims[2])});
      OpCommand("MatMul")
          .Input(tmp_x, "x1")
          .Input(Y, "x2")
          .Output(tmp_out, "y")
          .Attr("transpose_x1", trans_x)
          .Attr("transpose_x2", trans_y);
      OpCommand::Reshape(tmp_out, Out, out_dims);
      return;
    }

    // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
    std::vector<int> x_broadcast_dims(out_ndim, 1);
    std::vector<int> y_broadcast_dims(out_ndim, 1);
    std::copy(out_dims.begin(), out_dims.end() - 2, x_broadcast_dims.begin());
    std::copy(out_dims.begin(), out_dims.end() - 2, y_broadcast_dims.begin());
    std::copy(x_dims.end() - 2, x_dims.end(), x_broadcast_dims.end() - 2);
    std::copy(y_dims.end() - 2, y_dims.end(), y_broadcast_dims.end() - 2);

    bool broadcast_x = !(x_dims == x_broadcast_dims);
    bool broadcast_y = !(y_dims == y_broadcast_dims);

    Tensor x_brd, y_brd;
    if (broadcast_x) {
      OpCommand::BroadcastTo(X, x_brd, x_broadcast_dims);
    } else {
      x_brd = X;
    }
    if (broadcast_y) {
      OpCommand::BroadcastTo(Y, y_brd, y_broadcast_dims);
    } else {
      y_brd = Y;
    }
    OpCommand("BatchMatMul")
        .Input(x_brd, "x1")
        .Input(y_brd, "x2")
        .Output(Out, "y")
        .Attr("adj_x1", trans_x)
        .Attr("adj_x2", trans_y);
  }
};

class MatmulV2GradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& X = ctx.Input("X");
    auto& Y = ctx.Input("Y");
    auto& dOut = ctx.Input(paddle::framework::GradVarName("Out"));

    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    auto x_dims = X.Shape();
    auto y_dims = Y.Shape();
    auto out_dims = dOut.Shape();
    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();
    int out_ndim = out_dims.size();

    // Case 1: [K] x [K] = [1]
    if (x_ndim == 1 && y_ndim == 1) {
      if (ctx.HasOutput(paddle::framework::GradVarName("X"))) {
        auto& dX = ctx.Output(paddle::framework::GradVarName("X"));
        OpCommand("Mul").Input(dOut, "x1").Input(Y, "x2").Output(dX, "y");
      }
      if (ctx.HasOutput(paddle::framework::GradVarName("Y"))) {
        auto& dY = ctx.Output(paddle::framework::GradVarName("Y"));
        OpCommand("Mul").Input(dOut, "x1").Input(X, "x2").Output(dY, "y");
      }
      return;
    }

    Tensor x_temp, out_temp, y_temp;
    if (x_ndim == 1) {
      x_dims.insert(x_dims.begin(), 1);
      out_dims.insert(out_dims.end() - 1, 1);
      x_ndim = 2;
      out_ndim += 1;
      OpCommand::Reshape(X, x_temp, x_dims);
      y_temp = Y;
      OpCommand::Reshape(dOut, out_temp, out_dims);
    } else if (y_ndim == 1) {
      y_dims.push_back(1);
      out_dims.push_back(1);
      y_ndim = 2;
      out_ndim += 1;
      x_temp = X;
      OpCommand::Reshape(Y, y_temp, y_dims);
      OpCommand::Reshape(dOut, out_temp, out_dims);
    } else {
      x_temp = X;
      y_temp = Y;
      out_temp = dOut;
    }

    // Case 2: [M, K] x [K, N] = [M, N]
    if (out_ndim == 2) {
      if (ctx.HasOutput(paddle::framework::GradVarName("X"))) {
        auto& dX = ctx.Output(paddle::framework::GradVarName("X"));
        if (trans_x) {
          OpCommand("MatMul")
              .Input(y_temp, "x1")
              .Input(out_temp, "x2")
              .Output(dX)
              .Attr("transpose_x1", trans_y)
              .Attr("transpose_x2", true);
        } else {
          OpCommand("MatMul")
              .Input(out_temp, "x1")
              .Input(y_temp, "x2")
              .Output(dX)
              .Attr("transpose_x1", false)
              .Attr("transpose_x2", !trans_y);
        }
      }
      if (ctx.HasOutput(paddle::framework::GradVarName("Y"))) {
        auto& dY = ctx.Output(paddle::framework::GradVarName("Y"));
        if (trans_y) {
          OpCommand("MatMul")
              .Input(out_temp, "x1")
              .Input(x_temp, "x2")
              .Output(dY)
              .Attr("transpose_x1", true)
              .Attr("transpose_x2", trans_x);
        } else {
          OpCommand("MatMul")
              .Input(x_temp, "x1")
              .Input(out_temp, "x2")
              .Output(dY)
              .Attr("transpose_x1", !trans_x)
              .Attr("transpose_x2", false);
        }
      }
      return;
    }

    const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
    const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];

    // Case 3: [B, M, K] x [K, N] =  [B, M, N], when trans_x = false
    // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
    if (trans_x == false && y_ndim == 2) {
      Tensor dout_reshape;
      OpCommand::Reshape(out_temp, dout_reshape, {dOut.Numel() / N, N});
      if (ctx.HasOutput(paddle::framework::GradVarName("X"))) {
        auto& dX = ctx.Output(paddle::framework::GradVarName("X"));
        Tensor dx_reshape;
        OpCommand("MatMul")
            .Input(dout_reshape, "x1")
            .Input(y_temp, "x2")
            .Output(dx_reshape, "y")
            .Attr("transpose_x1", false)
            .Attr("transpose_x2", !trans_y);
        OpCommand::Reshape(dx_reshape, dX, X.Shape());
      }
      if (ctx.HasOutput(paddle::framework::GradVarName("Y"))) {
        auto& dY = ctx.Output(paddle::framework::GradVarName("Y"));
        Tensor x_temp_reshape;
        OpCommand::Reshape(x_temp, x_temp_reshape, {X.Numel() / K, K});
        if (trans_y) {
          OpCommand("MatMul")
              .Input(dout_reshape, "x1")
              .Input(x_temp_reshape, "x2")
              .Output(dY, "y")
              .Attr("transpose_x1", true)
              .Attr("transpose_x2", false);
        } else {
          OpCommand("MatMul")
              .Input(x_temp_reshape, "x1")
              .Input(dout_reshape, "x2")
              .Output(dY, "y")
              .Attr("transpose_x1", true)
              .Attr("transpose_x2", false);
        }
      }
      return;
    }

    // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
    std::vector<int> x_broadcast_dims(out_ndim, 1);
    std::vector<int> y_broadcast_dims(out_ndim, 1);
    std::copy(out_dims.begin(), out_dims.end() - 2, x_broadcast_dims.begin());
    std::copy(out_dims.begin(), out_dims.end() - 2, y_broadcast_dims.begin());
    std::copy(x_dims.end() - 2, x_dims.end(), x_broadcast_dims.end() - 2);
    std::copy(y_dims.end() - 2, y_dims.end(), y_broadcast_dims.end() - 2);

    Tensor x_temp_brd, y_temp_brd;
    if (x_dims == x_broadcast_dims) {
      OpCommand::Reshape(X, x_temp_brd, x_broadcast_dims);
    } else {
      OpCommand::BroadcastTo(X, x_temp_brd, x_broadcast_dims);
    }
    if (y_dims == y_broadcast_dims) {
      OpCommand::Reshape(Y, y_temp_brd, y_broadcast_dims);
    } else {
      OpCommand::BroadcastTo(Y, y_temp_brd, y_broadcast_dims);
    }

    if (ctx.HasOutput(paddle::framework::GradVarName("X"))) {
      auto& dX = ctx.Output(paddle::framework::GradVarName("X"));
      if (x_dims == x_broadcast_dims) {
        if (trans_x) {
          OpCommand("BatchMatMul")
              .Input(y_temp_brd, "x1")
              .Input(out_temp, "x2")
              .Output(dX, "y")
              .Attr("adj_x1", trans_y)
              .Attr("adj_x2", true);
        } else {
          OpCommand("BatchMatMul")
              .Input(out_temp, "x1")
              .Input(y_temp_brd, "x2")
              .Output(dX, "y")
              .Attr("adj_x1", false)
              .Attr("adj_x2", !trans_y);
        }
      } else {
        Tensor dx_temp;
        if (trans_x) {
          OpCommand("BatchMatMul")
              .Input(y_temp_brd, "x1")
              .Input(out_temp, "x2")
              .Output(dx_temp, "y")
              .Attr("adj_x1", trans_y)
              .Attr("adj_x2", true);
        } else {
          OpCommand("BatchMatMul")
              .Input(out_temp, "x1")
              .Input(y_temp_brd, "x2")
              .Output(dx_temp, "y")
              .Attr("adj_x1", false)
              .Attr("adj_x2", !trans_y);
        }

        std::vector<int64_t> axes;
        int64_t size = x_broadcast_dims.size();
        int64_t diff = x_broadcast_dims.size() - x_dims.size();
        for (int64_t i = 0; i < size; ++i) {
          if (i < diff) {
            axes.push_back(i);
            continue;
          }
          if (x_broadcast_dims[i] > x_dims[i - diff]) {
            axes.push_back(i);
          }
        }
        OpCommand("ReduceSumD")
            .Input(dx_temp)
            .Output(dX)
            .Attr("axes", axes)
            .Attr("keep_dims", false);
      }
    }
    if (ctx.HasOutput(paddle::framework::GradVarName("Y"))) {
      auto& dY = ctx.Output(paddle::framework::GradVarName("Y"));
      if (y_dims == y_broadcast_dims) {
        if (trans_y) {
          OpCommand("BatchMatMul")
              .Input(out_temp, "x1")
              .Input(x_temp_brd, "x2")
              .Output(dY, "y")
              .Attr("adj_x1", true)
              .Attr("adj_x2", trans_x);
        } else {
          OpCommand("BatchMatMul")
              .Input(x_temp_brd, "x1")
              .Input(out_temp, "x2")
              .Output(dY, "y")
              .Attr("adj_x1", !trans_x)
              .Attr("adj_x2", false);
        }
      } else {
        Tensor dy_temp;
        if (trans_y) {
          OpCommand("BatchMatMul")
              .Input(out_temp, "x1")
              .Input(x_temp_brd, "x2")
              .Output(dy_temp, "y")
              .Attr("adj_x1", true)
              .Attr("adj_x2", trans_x);
        } else {
          OpCommand("BatchMatMul")
              .Input(x_temp_brd, "x1")
              .Input(out_temp, "x2")
              .Output(dy_temp, "y")
              .Attr("adj_x1", !trans_x)
              .Attr("adj_x2", false);
        }

        std::vector<int64_t> axes;
        int64_t size = y_broadcast_dims.size();
        int64_t diff = y_broadcast_dims.size() - y_dims.size();
        for (int64_t i = 0; i < size; ++i) {
          if (i < diff) {
            axes.push_back(i);
            continue;
          }
          if (y_broadcast_dims[i] > y_dims[i - diff]) {
            axes.push_back(i);
          }
        }
        OpCommand("ReduceSumD")
            .Input(dy_temp)
            .Output(dY)
            .Attr("axes", axes)
            .Attr("keep_dims", false);
      }
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(matmul_v2, custom_graph::MatmulV2Adapter);
REG_OP_ADAPTER(matmul_v2_grad, custom_graph::MatmulV2GradAdapter);
