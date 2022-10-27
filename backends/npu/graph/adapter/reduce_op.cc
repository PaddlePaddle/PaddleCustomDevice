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

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& out = ctx.Output("Out");
    auto dim = ctx.Attr<std::vector<int>>("dim");
    bool reduce_all = ctx.Attr<bool>("reduce_all");
    bool keep_dim = ctx.Attr<bool>("keep_dim");
    auto x_dims = x.Shape();

    if (reduce_all) {
      dim.clear();
      for (auto i = 0; i < x_dims.size(); ++i) {
        dim.emplace_back(i);
      }
    }

    OpCommand("ReduceMeanD")
        .Input(x)
        .Output(out)
        .Attr("axes", std::vector<int64_t>(dim.begin(), dim.end()))
        .Attr("keep_dims", keep_dim);
  }
};

class ReduceMeanGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& out_grad = ctx.Input("Out@GRAD");
    auto& x_grad = ctx.Output("X@GRAD");
    auto axes = ctx.Attr<std::vector<int>>("dim");
    auto reduce_all = ctx.Attr<bool>("reduce_all");
    auto x_dim = x.Shape();

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

    Tensor muls, reshape_muls;
    OpCommand("Muls").Input(out_grad).Output(muls).Attr(
        "value", static_cast<float>(1.0f / reduce_numel));
    OpCommand::Reshape(muls, reshape_muls, out_dim);
    OpCommand::BroadcastTo(reshape_muls, x_grad, x_dim);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(reduce_mean, custom_graph::ReduceMeanAdapter);
REG_OP_ADAPTER(reduce_mean_grad, custom_graph::ReduceMeanGradAdapter);
