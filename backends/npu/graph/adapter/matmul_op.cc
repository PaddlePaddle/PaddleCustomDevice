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

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto* X = ctx.Input("X");
    auto* Y = ctx.Input("Y");
    auto* Out = ctx.Output("Out");
    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    auto x_dims = X->dims();
    auto y_dims = Y->dims();
    auto out_dims = Out->dims();
    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();
    int out_ndim = out_dims.size();

    if (x_ndim == 1 && y_ndim == 1) {
      //   PADDLE_ENFORCE_EQ(
      //       X->numel(),
      //       Y->numel(),
      //       platform::errors::InvalidArgument(
      //           "X's numbers must be equal to Y's numbers,"
      //           "when X/Y's dims =1. But received X has [%d] elements,"
      //           "received Y has [%d] elements",
      //           X->numel(),
      //           Y->numel()));
      auto node = ge::op::Dot()
                      .set_input_input_x(graph->GetOp(X->Name()))
                      .set_input_input_y(graph->GetOp(Y->Name()));
      graph::funcs::update_input_dtype(
          node, {{"input_x", X->dtype()}, {"input_y", Y->dtype()}});
      graph::funcs::update_output_dtype(node, {{"output", Out->dtype()}});
      graph->AddOp(Out->Name(), node);
      return;
    }

    // Case 2: [M, K] x [K, N] = [M, N]
    if (x_ndim == 2 && y_ndim == 2) {
      auto node = ge::op::MatMul()
                      .set_input_x1(graph->GetOp(X->Name()))
                      .set_input_x2(graph->GetOp(Y->Name()))
                      .set_attr_transpose_x1(trans_x)
                      .set_attr_transpose_x2(trans_y);
      graph::funcs::update_input_dtype(
          node, {{"x1", X->dtype()}, {"x2", Y->dtype()}});
      graph::funcs::update_output_dtype(node, {{"y", Out->dtype()}});
      graph->AddOp(Out->Name(), node);
      return;
    }

    // Case 3: [B, M, K] x [K, N] =  [B, M, N], when trans_x = false
    // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
    if (trans_x == false && y_ndim == 2) {
      auto reshape_x =
          graph::funcs::reshape(graph->GetOp(X->Name()),
                                {static_cast<int32_t>(x_dims[0] * x_dims[1]),
                                 static_cast<int32_t>(x_dims[2])});
      auto out = ge::op::MatMul()
                     .set_input_x1(reshape_x)
                     .set_input_x2(graph->GetOp(Y->Name()))
                     .set_attr_transpose_x1(trans_x)
                     .set_attr_transpose_x2(trans_y);
      auto reshape_out = graph::funcs::reshape(out, out_dims);
      graph::funcs::update_input_dtype(
          out, {{"x1", X->dtype()}, {"x2", Y->dtype()}});
      graph::funcs::update_output_dtype(out, {{"y", Out->dtype()}});
      graph->AddOp(Out->Name(), reshape_out);
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

    if (broadcast_x && broadcast_y) {
      auto x_broadcast =
          graph::funcs::broadcast_to(graph->GetOp(X->Name()), x_broadcast_dims);
      auto y_broadcast =
          graph::funcs::broadcast_to(graph->GetOp(X->Name()), y_broadcast_dims);
      auto node = ge::op::BatchMatMul()
                      .set_input_x1(x_broadcast)
                      .set_input_x2(y_broadcast)
                      .set_attr_adj_x1(trans_x)
                      .set_attr_adj_x2(trans_y);
      graph::funcs::update_input_dtype(
          node, {{"x1", X->dtype()}, {"x2", Y->dtype()}});
      graph::funcs::update_output_dtype(node, {{"y", Out->dtype()}});
      graph->AddOp(Out->Name(), node);
    } else if (broadcast_x && !broadcast_y) {
      auto x_broadcast =
          graph::funcs::broadcast_to(graph->GetOp(X->Name()), x_broadcast_dims);
      auto node = ge::op::BatchMatMul()
                      .set_input_x1(x_broadcast)
                      .set_input_x2(graph->GetOp(Y->Name()))
                      .set_attr_adj_x1(trans_x)
                      .set_attr_adj_x2(trans_y);
      graph::funcs::update_input_dtype(
          node, {{"x1", X->dtype()}, {"x2", Y->dtype()}});
      graph::funcs::update_output_dtype(node, {{"y", Out->dtype()}});
      graph->AddOp(Out->Name(), node);
    } else if (!broadcast_x && broadcast_y) {
      auto y_broadcast =
          graph::funcs::broadcast_to(graph->GetOp(X->Name()), y_broadcast_dims);
      auto node = ge::op::BatchMatMul()
                      .set_input_x1(graph->GetOp(X->Name()))
                      .set_input_x2(y_broadcast)
                      .set_attr_adj_x1(trans_x)
                      .set_attr_adj_x2(trans_y);
      graph::funcs::update_input_dtype(
          node, {{"x1", X->dtype()}, {"x2", Y->dtype()}});
      graph::funcs::update_output_dtype(node, {{"y", Out->dtype()}});
      graph->AddOp(Out->Name(), node);
    } else {
      auto node = ge::op::BatchMatMul()
                      .set_input_x1(graph->GetOp(X->Name()))
                      .set_input_x2(graph->GetOp(Y->Name()))
                      .set_attr_adj_x1(trans_x)
                      .set_attr_adj_x2(trans_y);
      graph::funcs::update_input_dtype(
          node, {{"x1", X->dtype()}, {"x2", Y->dtype()}});
      graph::funcs::update_output_dtype(node, {{"y", Out->dtype()}});
      graph->AddOp(Out->Name(), node);
    }
  }
};

class MatmulV2GradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto* X = ctx.Input("X");
    auto* Y = ctx.Input("Y");
    auto* dOut = ctx.Input(paddle::framework::GradVarName("Out"));
    auto* dX = ctx.Output(paddle::framework::GradVarName("X"));
    auto* dY = ctx.Output(paddle::framework::GradVarName("Y"));
    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    auto x_dims = X->dims();
    auto y_dims = Y->dims();
    auto out_dims = dOut->dims();
    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();
    int out_ndim = out_dims.size();

    // Case 1: [K] x [K] = [1]
    if (x_ndim == 1 && y_ndim == 1) {
      if (dX) {
        auto ge_op = ge::op::Mul()
                         .set_input_x1(graph->GetOp(dOut->Name()))
                         .set_input_x2(graph->GetOp(Y->Name()));
        graph::funcs::update_input_dtype(
            ge_op, {{"x1", dOut->dtype()}, {"x2", Y->dtype()}});
        graph::funcs::update_output_dtype(ge_op, {{"y", dX->dtype()}});
        graph->AddOp(dX->Name(), ge_op);
      }
      if (dY) {
        auto ge_op = ge::op::Mul()
                         .set_input_x1(graph->GetOp(dOut->Name()))
                         .set_input_x2(graph->GetOp(X->Name()));
        graph::funcs::update_input_dtype(
            ge_op, {{"x1", dOut->dtype()}, {"x2", X->dtype()}});
        graph::funcs::update_output_dtype(ge_op, {{"y", dY->dtype()}});
        graph->AddOp(dY->Name(), ge_op);
      }
      return;
    }

    if (x_ndim == 1) {
      x_dims.insert(x_dims.begin(), 1);
      out_dims.insert(out_dims.end() - 1, 1);
      x_ndim = 2;
      out_ndim += 1;
      auto x_temp = graph::funcs::reshape(graph->GetOp(X->Name()), x_dims);
      auto out_temp =
          graph::funcs::reshape(graph->GetOp(dOut->Name()), out_dims);
      graph->AddOp(X->Name() + "_temp", x_temp);
      graph->RecordNode(Y->Name() + "_temp", graph->GetOp(Y->Name()));
      graph->AddOp(dOut->Name() + "_temp", out_temp);
    } else if (y_ndim == 1) {
      y_dims.push_back(1);
      out_dims.push_back(1);
      y_ndim = 2;
      out_ndim += 1;
      auto y_temp = graph::funcs::reshape(graph->GetOp(Y->Name()), y_dims);
      auto out_temp =
          graph::funcs::reshape(graph->GetOp(dOut->Name()), out_dims);
      graph->RecordNode(X->Name() + "_temp", graph->GetOp(X->Name()));
      graph->AddOp(Y->Name() + "_temp", y_temp);
      graph->AddOp(dOut->Name() + "_temp", out_temp);
    } else {
      graph->RecordNode(X->Name() + "_temp", graph->GetOp(X->Name()));
      graph->RecordNode(Y->Name() + "_temp", graph->GetOp(Y->Name()));
      graph->RecordNode(dOut->Name() + "_temp", graph->GetOp(dOut->Name()));
    }

    // Case 2: [M, K] x [K, N] = [M, N]
    if (out_ndim == 2) {
      if (dX) {
        if (trans_x) {
          auto ge_op = ge::op::MatMul()
                           .set_input_x1(graph->GetOp(Y->Name() + "_temp"))
                           .set_input_x2(graph->GetOp(dOut->Name() + "_temp"))
                           .set_attr_transpose_x1(trans_y)
                           .set_attr_transpose_x2(true);
          graph::funcs::update_input_dtype(
              ge_op, {{"x1", Y->dtype()}, {"x2", dOut->dtype()}});
          graph::funcs::update_output_dtype(ge_op, {{"y", dX->dtype()}});
          graph->AddOp(dX->Name(), ge_op);
        } else {
          auto ge_op = ge::op::MatMul()
                           .set_input_x1(graph->GetOp(dOut->Name() + "_temp"))
                           .set_input_x2(graph->GetOp(Y->Name() + "_temp"))
                           .set_attr_transpose_x1(false)
                           .set_attr_transpose_x2(!trans_y);
          graph::funcs::update_input_dtype(
              ge_op, {{"x1", dOut->dtype()}, {"x2", Y->dtype()}});
          graph::funcs::update_output_dtype(ge_op, {{"y", dX->dtype()}});
          graph->AddOp(dX->Name(), ge_op);
        }
      }
      if (dY) {
        if (trans_y) {
          auto ge_op = ge::op::MatMul()
                           .set_input_x1(graph->GetOp(dOut->Name() + "_temp"))
                           .set_input_x2(graph->GetOp(X->Name() + "_temp"))
                           .set_attr_transpose_x1(true)
                           .set_attr_transpose_x2(trans_x);
          graph::funcs::update_input_dtype(
              ge_op, {{"x1", dOut->dtype()}, {"x2", X->dtype()}});
          graph::funcs::update_output_dtype(ge_op, {{"y", dY->dtype()}});
          graph->AddOp(dY->Name(), ge_op);
        } else {
          auto ge_op = ge::op::MatMul()
                           .set_input_x1(graph->GetOp(X->Name() + "_temp"))
                           .set_input_x2(graph->GetOp(dOut->Name() + "_temp"))
                           .set_attr_transpose_x1(!trans_x)
                           .set_attr_transpose_x2(false);
          graph::funcs::update_input_dtype(
              ge_op, {{"x1", X->dtype()}, {"x2", dOut->dtype()}});
          graph::funcs::update_output_dtype(ge_op, {{"y", dY->dtype()}});
          graph->AddOp(dY->Name(), ge_op);
        }
      }
      return;
    }

    const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
    const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];

    // Case 3: [B, M, K] x [K, N] =  [B, M, N], when trans_x = false
    // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
    if (trans_x == false && y_ndim == 2) {
      auto dout_reshape = graph::funcs::reshape(
          graph->GetOp(dOut->Name() + "_temp"), {dOut->numel() / N, N});

      if (dX) {
        auto matmul = ge::op::MatMul()
                          .set_input_x1(dout_reshape)
                          .set_input_x2(graph->GetOp(Y->Name() + "_temp"))
                          .set_attr_transpose_x1(false)
                          .set_attr_transpose_x2(!trans_y);
        auto matmul_reshape = graph::funcs::reshape(matmul, X->dims());
        graph::funcs::update_input_dtype(
            matmul, {{"x1", dOut->dtype()}, {"x2", Y->dtype()}});
        graph::funcs::update_output_dtype(matmul, {{"y", dX->dtype()}});
        graph->AddOp(dX->Name(), matmul_reshape);
      }
      if (dY) {
        auto x_temp_reshape = graph::funcs::reshape(
            graph->GetOp(X->Name() + "_temp"), {X->numel() / K, K});
        if (trans_y) {
          auto matmul = ge::op::MatMul()
                            .set_input_x1(dout_reshape)
                            .set_input_x2(x_temp_reshape)
                            .set_attr_transpose_x1(true)
                            .set_attr_transpose_x2(false);
          graph::funcs::update_input_dtype(
              matmul, {{"x1", dOut->dtype()}, {"x2", X->dtype()}});
          graph::funcs::update_output_dtype(matmul, {{"y", dY->dtype()}});
          graph->AddOp(dY->Name(), matmul);
        } else {
          auto matmul = ge::op::MatMul()
                            .set_input_x1(x_temp_reshape)
                            .set_input_x2(dout_reshape)
                            .set_attr_transpose_x1(true)
                            .set_attr_transpose_x2(false);
          graph::funcs::update_input_dtype(
              matmul, {{"x1", X->dtype()}, {"x2", dOut->dtype()}});
          graph::funcs::update_output_dtype(matmul, {{"y", dY->dtype()}});
          graph->AddOp(dY->Name(), matmul);
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

    if (x_dims == x_broadcast_dims) {
      auto x_temp_brd =
          graph::funcs::reshape(graph->GetOp(X->Name()), x_broadcast_dims);
      graph->AddOp(X->Name() + "_temp_brd", x_temp_brd);
    } else {
      auto x_temp_brd = graph::funcs::broadcast_to(
          graph->GetOp(X->Name() + "_temp"), x_broadcast_dims);
      graph->AddOp(X->Name() + "_temp_brd", x_temp_brd);
    }

    if (y_dims == y_broadcast_dims) {
      auto y_temp_brd =
          graph::funcs::reshape(graph->GetOp(Y->Name()), y_broadcast_dims);
      graph->AddOp(X->Name() + "_temp_brd", y_temp_brd);
    } else {
      auto y_temp_brd = graph::funcs::broadcast_to(
          graph->GetOp(Y->Name() + "_temp"), y_broadcast_dims);
      graph->AddOp(Y->Name() + "_temp_brd", y_temp_brd);
    }

    if (dX) {
      if (x_dims == x_broadcast_dims) {
        if (trans_x) {
          auto ge_op = ge::op::BatchMatMul()
                           .set_input_x1(graph->GetOp(Y->Name() + "_temp_brd"))
                           .set_input_x2(graph->GetOp(dOut->Name() + "_temp"))
                           .set_attr_adj_x1(trans_y)
                           .set_attr_adj_x2(true);
          graph::funcs::update_input_dtype(
              ge_op, {{"x1", Y->dtype()}, {"x2", dOut->dtype()}});
          graph::funcs::update_output_dtype(ge_op, {{"y", dX->dtype()}});
          graph->AddOp(dX->Name(), ge_op);
        } else {
          auto ge_op = ge::op::BatchMatMul()
                           .set_input_x1(graph->GetOp(dOut->Name() + "_temp"))
                           .set_input_x2(graph->GetOp(Y->Name() + "_temp_brd"))
                           .set_attr_adj_x1(false)
                           .set_attr_adj_x2(!trans_y);
          graph::funcs::update_input_dtype(
              ge_op, {{"x1", dOut->dtype()}, {"x2", Y->dtype()}});
          graph::funcs::update_output_dtype(ge_op, {{"y", dX->dtype()}});
          graph->AddOp(dX->Name(), ge_op);
        }
      } else {
        if (trans_x) {
          auto ge_op = ge::op::BatchMatMul()
                           .set_input_x1(graph->GetOp(Y->Name() + "_temp_brd"))
                           .set_input_x2(graph->GetOp(dOut->Name() + "_temp"))
                           .set_attr_adj_x1(trans_y)
                           .set_attr_adj_x2(true);
          graph::funcs::update_input_dtype(
              ge_op, {{"x1", Y->dtype()}, {"x2", dOut->dtype()}});
          graph::funcs::update_output_dtype(ge_op, {{"y", dX->dtype()}});
          graph->AddOp(dX->Name() + "_temp", ge_op);
        } else {
          auto ge_op = ge::op::BatchMatMul()
                           .set_input_x1(graph->GetOp(dOut->Name() + "_temp"))
                           .set_input_x2(graph->GetOp(Y->Name() + "_temp_brd"))
                           .set_attr_adj_x1(false)
                           .set_attr_adj_x2(!trans_y);
          graph::funcs::update_input_dtype(
              ge_op, {{"x1", dOut->dtype()}, {"x2", Y->dtype()}});
          graph::funcs::update_output_dtype(ge_op, {{"y", dX->dtype()}});
          graph->AddOp(dX->Name() + "_temp", ge_op);
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
        auto ge_op = ge::op::ReduceSumD()
                         .set_input_x(graph->GetOp(dX->Name() + "_temp"))
                         .set_attr_axes(axes)
                         .set_attr_keep_dims(false);
        graph->AddOp(dX->Name(), ge_op);
      }
    }
    if (dY) {
      if (y_dims == y_broadcast_dims) {
        if (trans_y) {
          auto ge_op = ge::op::BatchMatMul()
                           .set_input_x1(graph->GetOp(dOut->Name() + "_temp"))
                           .set_input_x2(graph->GetOp(X->Name() + "_temp_brd"))
                           .set_attr_adj_x1(true)
                           .set_attr_adj_x2(trans_x);
          graph::funcs::update_input_dtype(
              ge_op, {{"x1", dOut->dtype()}, {"x2", X->dtype()}});
          graph::funcs::update_output_dtype(ge_op, {{"y", dY->dtype()}});
          graph->AddOp(dY->Name(), ge_op);
        } else {
          auto ge_op = ge::op::BatchMatMul()
                           .set_input_x1(graph->GetOp(X->Name() + "_temp_brd"))
                           .set_input_x2(graph->GetOp(dOut->Name() + "_temp"))
                           .set_attr_adj_x1(!trans_x)
                           .set_attr_adj_x2(false);
          graph::funcs::update_input_dtype(
              ge_op, {{"x1", X->dtype()}, {"x2", dOut->dtype()}});
          graph::funcs::update_output_dtype(ge_op, {{"y", dY->dtype()}});
          graph->AddOp(dY->Name(), ge_op);
        }
      } else {
        if (trans_y) {
          auto ge_op = ge::op::BatchMatMul()
                           .set_input_x1(graph->GetOp(dOut->Name() + "_temp"))
                           .set_input_x2(graph->GetOp(X->Name() + "_temp_brd"))
                           .set_attr_adj_x1(true)
                           .set_attr_adj_x2(trans_x);
          graph::funcs::update_input_dtype(
              ge_op, {{"x1", dOut->dtype()}, {"x2", X->dtype()}});
          graph::funcs::update_output_dtype(ge_op, {{"y", dY->dtype()}});
          graph->AddOp(dY->Name() + "_temp", ge_op);
        } else {
          auto ge_op = ge::op::BatchMatMul()
                           .set_input_x1(graph->GetOp(X->Name() + "_temp_brd"))
                           .set_input_x2(graph->GetOp(dOut->Name() + "_temp"))
                           .set_attr_adj_x1(!trans_x)
                           .set_attr_adj_x2(false);
          graph::funcs::update_input_dtype(
              ge_op, {{"x1", X->dtype()}, {"x2", dOut->dtype()}});
          graph::funcs::update_output_dtype(ge_op, {{"y", dY->dtype()}});
          graph->AddOp(dY->Name() + "_temp", ge_op);
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
        auto ge_op = ge::op::ReduceSumD()
                         .set_input_x(graph->GetOp(dY->Name() + "_temp"))
                         .set_attr_axes(axes)
                         .set_attr_keep_dims(false);
        graph->AddOp(dY->Name(), ge_op);
      }
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(matmul_v2, custom_graph::MatmulV2Adapter);
REG_OP_ADAPTER(matmul_v2_grad, custom_graph::MatmulV2GradAdapter);
