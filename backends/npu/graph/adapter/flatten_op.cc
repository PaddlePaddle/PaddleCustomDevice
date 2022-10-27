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

class FlattenContiguousRangeAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& out = ctx.Output("Out");
    int start_axis = ctx.Attr<int>("start_axis");
    int stop_axis = ctx.Attr<int>("stop_axis");

    OpCommand("FlattenV2")
        .Input(x)
        .Output(out)
        .Attr("axis", start_axis)
        .Attr("end_axis", stop_axis);
  }
};

class FlattenContiguousRangeGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& out_grad = ctx.Input("Out@GRAD");
    auto& x_grad = ctx.Output("X@GRAD");

    auto xshape_dims = ctx.Input("XShape").Shape();
    auto x_dims = slice_ddim(xshape_dims, 1, xshape_dims.size());

    OpCommand::Reshape(out_grad, x_grad, x_dims);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(flatten_contiguous_range,
               custom_graph::FlattenContiguousRangeAdapter);
REG_OP_ADAPTER(flatten_contiguous_range_grad,
               custom_graph::FlattenContiguousRangeGradAdapter);
