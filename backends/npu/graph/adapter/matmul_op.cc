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
      graph->AddOp(Out->Name(), node);
      return;
    }

    // Case 3: [B, M, K] x [K, N] =  [B, M, N], when trans_x = false
    // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
    if (trans_x == false && y_ndim == 2) {
      auto reshape_x_shape = graph::funcs::constant(
          {2},
          std::vector<int32_t>({static_cast<int32_t>(x_dims[0] * x_dims[1]),
                                static_cast<int32_t>(x_dims[2])}));
      auto reshape_x = ge::op::Reshape()
                           .set_input_x(graph->GetOp(X->Name()))
                           .set_input_shape(reshape_x_shape);

      auto out = ge::op::MatMul()
                     .set_input_x1(reshape_x)
                     .set_input_x2(graph->GetOp(Y->Name()))
                     .set_attr_transpose_x1(trans_x)
                     .set_attr_transpose_x2(trans_y);

      auto reshape_out_shape = graph::funcs::constant({3}, out_dims);
      auto reshape_out =
          ge::op::Reshape().set_input_x(out).set_input_shape(reshape_out_shape);

      graph->AddOp(Out->Name(), reshape_out);
      return;
    }

    // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
    std::vector<int64_t> x_broadcast_dims(out_ndim, 1);
    std::vector<int64_t> y_broadcast_dims(out_ndim, 1);
    std::copy(out_dims.begin(), out_dims.end() - 2, x_broadcast_dims.begin());
    std::copy(out_dims.begin(), out_dims.end() - 2, y_broadcast_dims.begin());
    std::copy(x_dims.end() - 2, x_dims.end(), x_broadcast_dims.end() - 2);
    std::copy(y_dims.end() - 2, y_dims.end(), y_broadcast_dims.end() - 2);

    bool broadcast_x =
        !(x_dims.size() == x_broadcast_dims.size() &&
          std::equal(x_dims.begin(), x_dims.end(), x_broadcast_dims.begin()));
    bool broadcast_y =
        !(y_dims.size() == y_broadcast_dims.size() &&
          std::equal(y_dims.begin(), y_dims.end(), y_broadcast_dims.begin()));

    if (broadcast_x && broadcast_y) {
      auto x_broadcast_shape =
          graph::funcs::constant({x_broadcast_dims.size()}, x_broadcast_dims);
      auto y_broadcast_shape =
          graph::funcs::constant({y_broadcast_dims.size()}, y_broadcast_dims);
      auto x_broadcast = ge::op::BroadcastTo()
                             .set_input_x(graph->GetOp(X->Name()))
                             .set_input_shape(x_broadcast_shape);
      auto y_broadcast = ge::op::BroadcastTo()
                             .set_input_x(graph->GetOp(Y->Name()))
                             .set_input_shape(y_broadcast_shape);
      auto node = ge::op::BatchMatMul()
                      .set_input_x1(x_broadcast)
                      .set_input_x2(y_broadcast)
                      .set_attr_adj_x1(trans_x)
                      .set_attr_adj_x2(trans_y);
      graph->AddOp(Out->Name(), node);
    } else if (broadcast_x && !broadcast_y) {
      auto x_broadcast_shape =
          graph::funcs::constant({x_broadcast_dims.size()}, x_broadcast_dims);
      auto x_broadcast = ge::op::BroadcastTo()
                             .set_input_x(graph->GetOp(X->Name()))
                             .set_input_shape(x_broadcast_shape);
      auto node = ge::op::BatchMatMul()
                      .set_input_x1(x_broadcast)
                      .set_input_x2(graph->GetOp(Y->Name()))
                      .set_attr_adj_x1(trans_x)
                      .set_attr_adj_x2(trans_y);
      graph->AddOp(Out->Name(), node);
    } else if (!broadcast_x && broadcast_y) {
      auto y_broadcast_shape =
          graph::funcs::constant({y_broadcast_dims.size()}, y_broadcast_dims);
      auto y_broadcast = ge::op::BroadcastTo()
                             .set_input_x(graph->GetOp(Y->Name()))
                             .set_input_shape(y_broadcast_shape);
      auto node = ge::op::BatchMatMul()
                      .set_input_x1(graph->GetOp(X->Name()))
                      .set_input_x2(y_broadcast)
                      .set_attr_adj_x1(trans_x)
                      .set_attr_adj_x2(trans_y);
      graph->AddOp(Out->Name(), node);
    } else {
      auto node = ge::op::BatchMatMul()
                      .set_input_x1(graph->GetOp(X->Name()))
                      .set_input_x2(graph->GetOp(Y->Name()))
                      .set_attr_adj_x1(trans_x)
                      .set_attr_adj_x2(trans_y);
      graph->AddOp(Out->Name(), node);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(matmul_v2, custom_graph::MatmulV2Adapter);
