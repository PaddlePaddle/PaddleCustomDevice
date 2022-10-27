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

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& y = ctx.Input("Y");
    auto& out = ctx.Output("Out");

    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");

    auto x_dim = x.Shape();
    auto y_dim = y.Shape();

    if (x_num_col_dims == 1 && y_num_col_dims == 1) {
      if (x_dim.size() == 2 && y_dim.size() == 2) {
        OpCommand("MatMul")
            .Input(x)
            .Input(y)
            .Output(out)
            .Attr("transpose_x1", false)
            .Attr("transpose_x2", false);
      } else if (x_dim.size() >= 3 && y_dim.size() == 2) {
        OpCommand("BatchMatMul")
            .Input(x)
            .Input(y)
            .Output(out)
            .Attr("adj_x1", false)
            .Attr("adj_x2", false);
      } else {
        // error
      }
    } else if (x_dim.size() == 3 && y_dim.size() == 2) {
      if (x_num_col_dims == 2) {
        OpCommand("BatchMatMul")
            .Input(x)
            .Input(y)
            .Output(out)
            .Attr("adj_x1", false)
            .Attr("adj_x2", false);
      } else {
        // error
      }
    }
  }
};

class MulGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& y = ctx.Input("Y");
    auto& out_grad = ctx.Input("Out@GRAD");

    auto x_dim = x.Shape();
    auto y_dim = y.Shape();

    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");

    if (x_num_col_dims == 1 && y_num_col_dims == 1) {
      if (x_dim.size() == 2 && y_dim.size() == 2) {
        if (ctx.HasOutput("X@GRAD")) {
          auto& x_grad = ctx.Output("X@GRAD");
          OpCommand("MatMul")
              .Input(out_grad)
              .Input(y)
              .Output(x_grad)
              .Attr("transpose_x1", false)
              .Attr("transpose_x2", true);
        }
        if (ctx.HasOutput("Y@GRAD")) {
          auto& y_grad = ctx.Output("Y@GRAD");
          OpCommand("MatMul")
              .Input(x)
              .Input(out_grad)
              .Output(y_grad)
              .Attr("transpose_x1", true)
              .Attr("transpose_x2", false);
        }
      } else if (x_dim.size() >= 3 && y_dim.size() == 2) {
        if (ctx.HasOutput("X@GRAD")) {
          auto& x_grad = ctx.Output("X@GRAD");
          OpCommand("BatchMatMul")
              .Input(out_grad)
              .Input(y)
              .Output(x_grad)
              .Attr("adj_x1", false)
              .Attr("adj_x2", true);
        }
        if (ctx.HasOutput("Y@GRAD")) {
          auto& y_grad = ctx.Output("Y@GRAD");
          OpCommand("BatchMatMul")
              .Input(x)
              .Input(out_grad)
              .Output(y_grad)
              .Attr("adj_x1", true)
              .Attr("adj_x2", false);
        }
      }
    } else {
      if (x_num_col_dims == 2) {
        if (ctx.HasOutput("X@GRAD")) {
          auto& x_grad = ctx.Output("X@GRAD");
          OpCommand("BatchMatMul")
              .Input(out_grad)
              .Input(y)
              .Output(x_grad)
              .Attr("adj_x1", false)
              .Attr("adj_x2", true);
        }
        if (ctx.HasOutput("Y@GRAD")) {
          auto& y_grad = ctx.Output("Y@GRAD");
          OpCommand("BatchMatMul")
              .Input(x)
              .Input(out_grad)
              .Output(y_grad)
              .Attr("adj_x1", true)
              .Attr("adj_x2", false);
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
