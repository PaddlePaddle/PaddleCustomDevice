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

class ReduceMeanAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto x = ctx.Input("X");
    auto out = ctx.Output("Out");

    auto x_dims = ctx.Input("X")->dims();
    auto dim = ctx.Attr<std::vector<int>>("dim");
    bool reduce_all = ctx.Attr<bool>("reduce_all");
    bool keep_dim = ctx.Attr<bool>("keep_dim");

    if (reduce_all) {
      dim.clear();
      for (auto i = 0; i < x_dims.size(); ++i) {
        dim.emplace_back(i);
      }
    }
    auto ge_op =
        ge::op::ReduceMeanD(ge::AscendString(out->Name().c_str()))
            .set_input_x(graph->GetOp(x->Name()))
            .set_attr_axes(std::vector<int64_t>(dim.begin(), dim.end()))
            .set_attr_keep_dims(keep_dim);
    graph->AddOp(out->Name(), ge_op);
  }
};

class ReduceMeanGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto x = ctx.Input("X");
    auto x_dim = x->dims();
    auto out_grad = ctx.Input("Out@GRAD");
    auto x_grad = ctx.Output("X@GRAD");
    auto axes = ctx.Attr<std::vector<int>>("dim");
    auto reduce_all = ctx.Attr<bool>("reduce_all");

    int reduce_numel = 1;
    if (reduce_all) {
      axes.clear();
      for (auto i = 0; i < x_dim.size(); ++i) {
        axes.push_back(i);
      }
    }

    std::vector<int> out_dim = x_dim;
    for (auto i = 0; i < axes.size(); ++i) {
      if (axes[i] < 0) {
        axes[i] = x_dim.size() + axes[i];
      }
      reduce_numel *= x_dim[axes[i]];
      out_dim[axes[i]] = 1;
    }
    auto fillv2 = ge::op::FillV2D()
                      .set_attr_value(1.0f / static_cast<float>(reduce_numel))
                      .set_attr_dims(
                          std::vector<int64_t>(out_dim.begin(), out_dim.end()));
    auto mul = ge::op::Mul().set_input_x1(fillv2).set_input_x2(
        graph->GetOp(out_grad->Name()));
    auto zeros_like_x = ge::op::FillV2D().set_attr_value(0.0f).set_attr_dims(
        std::vector<int64_t>(x_dim.begin(), x_dim.end()));
    auto add = ge::op::Add(ge::AscendString((x->Name() + "_grad").c_str()))
                   .set_input_x1(mul)
                   .set_input_x2(zeros_like_x);
    graph->AddOp(x_grad->Name(), add);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(reduce_mean, custom_graph::ReduceMeanAdapter);
REG_OP_ADAPTER(reduce_mean_grad, custom_graph::ReduceMeanGradAdapter);
