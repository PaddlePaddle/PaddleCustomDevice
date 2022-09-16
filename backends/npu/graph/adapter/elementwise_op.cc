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

class ElementwiseAddAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto out = ctx.Output("Out");
    auto x = ctx.Input("X");
    auto y = ctx.Input("Y");
    auto ge_op = ge::op::Add(ge::AscendString(out->Name().c_str()))
                     .set_input_x1(graph->GetOp(x->Name()))
                     .set_input_x2(graph->GetOp(y->Name()));
    graph->AddOp(out->Name(), ge_op);
  }
};

class ElementwiseAddGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto x = ctx.Input("X");
    auto x_dim = x->dims();
    auto y = ctx.Input("Y");
    auto y_dim = y->dims();
    auto out_grad = ctx.Input("Out@GRAD");
    auto out_grad_dim = out_grad->dims();
    auto x_grad = ctx.Output("X@GRAD");
    auto x_grad_dim = x_grad->dims();
    auto y_grad = ctx.Output("Y@GRAD");
    auto y_grad_dim = y_grad->dims();

    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(static_cast<int>(x_dim.size()) -
                                  static_cast<int>(y_dim.size()))
                       : axis);

    if (x_grad) {
      if (x_dim.size() == out_grad_dim.size() &&
          std::equal(x_dim.begin(), x_dim.end(), out_grad_dim.begin())) {
        auto var_dims = x_grad->dims();
        ge::TensorDesc var_desc(
            ge::Shape(std::vector<int64_t>(var_dims.begin(), var_dims.end())),
            ge::Format::FORMAT_NCHW,
            graph::traits::pd_dtype_to_ge_dtype(x_grad->dtype()));
        var_desc.SetRealDimCnt(var_desc.GetShape().GetDimNum());

        auto var =
            ge::op::Variable(ge::AscendString((x->Name() + "_grad").c_str()));
        var.update_output_desc_y(var_desc);
        auto assign_op = ge::op::Assign().set_input_ref(var).set_input_value(
            graph->GetOp(out_grad->Name()));
        graph->AddOp(x_grad->Name(), var);
        graph->AddInput(graph->GetOp(x_grad->Name()));
      } else {
        std::vector<int64_t> reduce_axes;
        for (auto i = 0; i < out_grad_dim.size(); ++i) {
          if (i < axis || i > axis + x_dim.size() ||
              (x_dim[i] == 1 && out_grad_dim[i] > 1)) {
            reduce_axes.push_back(i);
          }
        }
        if (reduce_axes.size() > 0) {
          auto ge_op = ge::op::ReduceSumD(
                           ge::AscendString((x->Name() + "_grad").c_str()))
                           .set_input_x(graph->GetOp(out_grad->Name()))
                           .set_attr_axes(reduce_axes)
                           .set_attr_keep_dims(false);
          graph->AddOp(x_grad->Name(), ge_op);
        } else {
          // error
        }
      }
    }
    if (y_grad) {
      if (y_dim.size() == out_grad_dim.size() &&
          std::equal(y_dim.begin(), y_dim.end(), out_grad_dim.begin())) {
        auto var_dims = y_grad->dims();
        ge::TensorDesc var_desc(
            ge::Shape(std::vector<int64_t>(var_dims.begin(), var_dims.end())),
            ge::Format::FORMAT_NCHW,
            graph::traits::pd_dtype_to_ge_dtype(y_grad->dtype()));
        var_desc.SetRealDimCnt(var_desc.GetShape().GetDimNum());

        auto var =
            ge::op::Variable(ge::AscendString((y->Name() + "_grad").c_str()));
        var.update_output_desc_y(var_desc);
        auto assign_op = ge::op::Assign().set_input_ref(var).set_input_value(
            graph->GetOp(out_grad->Name()));
        graph->AddOp(y_grad->Name(), var);
        graph->AddInput(graph->GetOp(y_grad->Name()));
      } else {
        std::vector<int64_t> reduce_axes;
        for (auto i = 0; i < out_grad_dim.size(); ++i) {
          if (i < axis || i > axis + y_dim.size() ||
              (y_dim[i] == 1 && out_grad_dim[i] > 1)) {
            reduce_axes.push_back(i);
          }
        }
        if (reduce_axes.size() > 0) {
          auto ge_op = ge::op::ReduceSumD(
                           ge::AscendString((y->Name() + "_grad").c_str()))
                           .set_input_x(graph->GetOp(out_grad->Name()))
                           .set_attr_axes(reduce_axes)
                           .set_attr_keep_dims(false);
          graph->AddOp(y_grad->Name(), ge_op);
        } else {
          // error
        }
      }
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(elementwise_add, custom_graph::ElementwiseAddAdapter);
REG_OP_ADAPTER(elementwise_add_grad, custom_graph::ElementwiseAddGradAdapter);
