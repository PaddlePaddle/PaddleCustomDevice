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

class MulAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto x = ctx.Input("X");
    auto x_dim = x->dims();
    auto y = ctx.Input("Y");
    auto y_dim = y->dims();
    auto out = ctx.Output("Out");
    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");

    if (x_num_col_dims == 1 && y_num_col_dims == 1) {
      if (x_dim.size() == 2 && y_dim.size() == 2) {
        auto ge_op = ge::op::MatMul(ge::AscendString(out->Name().c_str()))
                         .set_input_x1(graph->GetOp(x->Name()))
                         .set_input_x2(graph->GetOp(y->Name()))
                         .set_attr_transpose_x1(false)
                         .set_attr_transpose_x2(false);
        graph->AddOp(out->Name(), ge_op);
      } else if (x_dim.size() >= 3 && y_dim.size() == 2) {
        auto ge_op = ge::op::BatchMatMul(ge::AscendString(out->Name().c_str()))
                         .set_input_x1(graph->GetOp(x->Name()))
                         .set_input_x2(graph->GetOp(y->Name()))
                         .set_attr_adj_x1(false)
                         .set_attr_adj_x2(false);
        graph->AddOp(out->Name(), ge_op);
      } else {
        // error
      }
    } else if (x_dim.size() == 3 && y_dim.size() == 2) {
      if (x_num_col_dims == 2) {
        auto ge_op = ge::op::BatchMatMul(ge::AscendString(out->Name().c_str()))
                         .set_input_x1(graph->GetOp(x->Name()))
                         .set_input_x2(graph->GetOp(y->Name()))
                         .set_attr_adj_x1(false)
                         .set_attr_adj_x2(false);
        graph->AddOp(out->Name(), ge_op);
      } else {
        // error
      }
    }
  }
};

class MulGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto x = ctx.Input("X");
    auto x_dim = x->dims();
    auto y = ctx.Input("Y");
    auto y_dim = y->dims();
    auto out_grad = ctx.Input("Out@GRAD");
    auto x_grad = ctx.Output("X@GRAD");
    auto y_grad = ctx.Output("Y@GRAD");

    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");

    if (x_num_col_dims == 1 && y_num_col_dims == 1) {
      if (x_dim.size() == 2 && y_dim.size() == 2) {
        if (x_grad) {
          auto ge_op =
              ge::op::MatMul(ge::AscendString((x->Name() + "_grad").c_str()))
                  .set_input_x1(graph->GetOp(out_grad->Name()))
                  .set_input_x2(graph->GetOp(y->Name()))
                  .set_attr_transpose_x1(false)
                  .set_attr_transpose_x2(true);
          graph->AddOp(x_grad->Name(), ge_op);
        }
        if (y_grad) {
          auto ge_op =
              ge::op::MatMul(ge::AscendString((y->Name() + "_grad").c_str()))
                  .set_input_x1(graph->GetOp(x->Name()))
                  .set_input_x2(graph->GetOp(out_grad->Name()))
                  .set_attr_transpose_x1(true)
                  .set_attr_transpose_x2(false);
          graph->AddOp(y_grad->Name(), ge_op);
        }
      } else if (x_dim.size() >= 3 && y_dim.size() == 2) {
        if (x_grad) {
          auto ge_op = ge::op::BatchMatMul(
                           ge::AscendString((x->Name() + "_grad").c_str()))
                           .set_input_x1(graph->GetOp(out_grad->Name()))
                           .set_input_x2(graph->GetOp(y->Name()))
                           .set_attr_adj_x1(false)
                           .set_attr_adj_x2(true);
          graph->AddOp(x_grad->Name(), ge_op);
        }
        if (y_grad) {
          auto ge_op = ge::op::BatchMatMul(
                           ge::AscendString((y->Name() + "_grad").c_str()))
                           .set_input_x1(graph->GetOp(x->Name()))
                           .set_input_x2(graph->GetOp(out_grad->Name()))
                           .set_attr_adj_x1(true)
                           .set_attr_adj_x2(false);
          graph->AddOp(y_grad->Name(), ge_op);
        }
      }
    } else {
      if (x_num_col_dims == 2) {
        if (x_grad) {
          auto ge_op = ge::op::BatchMatMul(
                           ge::AscendString((x->Name() + "_grad").c_str()))
                           .set_input_x1(graph->GetOp(out_grad->Name()))
                           .set_input_x2(graph->GetOp(y->Name()))
                           .set_attr_adj_x1(false)
                           .set_attr_adj_x2(true);
          graph->AddOp(x_grad->Name(), ge_op);
        }
        if (y_grad) {
          auto ge_op = ge::op::BatchMatMul(
                           ge::AscendString((y->Name() + "_grad").c_str()))
                           .set_input_x1(graph->GetOp(x->Name()))
                           .set_input_x2(graph->GetOp(out_grad->Name()))
                           .set_attr_adj_x1(true)
                           .set_attr_adj_x2(false);
          graph->AddOp(y_grad->Name(), ge_op);
        }
      } else {
        // error
      }
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(mul, custom_graph::MulAdapter);
REG_OP_ADAPTER(mul_grad, custom_graph::MulGradAdapter);
