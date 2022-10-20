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
    auto x_dims = x->dims();
    auto y_dims = y->dims();

    auto axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(static_cast<int>(x_dims.size()) -
                                  static_cast<int>(y_dims.size()))
                       : axis);

    if (x_dims.size() >= y_dims.size() &&
        y_dims.size() + axis < x_dims.size()) {
      if (x_dims.size() == 4 && y_dims.size() == 1 && axis == 3) {
        auto ge_op = ge::op::BiasAdd(ge::AscendString(out->Name().c_str()))
                         .set_input_x(graph->GetOp(x->Name()))
                         .set_input_bias(graph->GetOp(y->Name()))
                         .set_attr_data_format("NHWC");
        graph->AddOp(out->Name(), ge_op);
      } else if (x_dims.size() == 4 && y_dims.size() == 1 && axis == 1) {
        auto ge_op = ge::op::BiasAdd(ge::AscendString(out->Name().c_str()))
                         .set_input_x(graph->GetOp(x->Name()))
                         .set_input_bias(graph->GetOp(y->Name()))
                         .set_attr_data_format("NCHW");
        graph->AddOp(out->Name(), ge_op);
      } else {
        auto y_dims_tmp = y_dims;
        for (auto i = 0; i < x_dims.size() - y_dims.size() - axis; ++i) {
          y_dims_tmp.push_back(1);
        }
        auto reshape_y =
            graph::funcs::reshape(graph->GetOp(y->Name()), y_dims_tmp);
        auto ge_op = ge::op::Add(ge::AscendString(out->Name().c_str()))
                         .set_input_x1(graph->GetOp(x->Name()))
                         .set_input_x2(reshape_y);
        graph->AddOp(out->Name(), ge_op);
      }
    } else if (x_dims.size() < y_dims.size() &&
               x_dims.size() + axis < y_dims.size()) {
      if (y_dims.size() == 4 && x_dims.size() == 1 && axis == 3) {
        auto ge_op = ge::op::BiasAdd(ge::AscendString(out->Name().c_str()))
                         .set_input_x(graph->GetOp(y->Name()))
                         .set_input_bias(graph->GetOp(x->Name()))
                         .set_attr_data_format("NHWC");
        graph->AddOp(out->Name(), ge_op);
      } else if (y_dims.size() == 4 && x_dims.size() == 1 && axis == 1) {
        auto ge_op = ge::op::BiasAdd(ge::AscendString(out->Name().c_str()))
                         .set_input_x(graph->GetOp(y->Name()))
                         .set_input_bias(graph->GetOp(x->Name()))
                         .set_attr_data_format("NCHW");
        graph->AddOp(out->Name(), ge_op);
      } else {
        auto x_dims_tmp = x_dims;
        for (auto i = 0; i < y_dims.size() - x_dims.size() - axis; ++i) {
          x_dims_tmp.push_back(1);
        }
        auto reshape_x =
            graph::funcs::reshape(graph->GetOp(x->Name()), x_dims_tmp);
        auto ge_op = ge::op::Add(ge::AscendString(out->Name().c_str()))
                         .set_input_x1(reshape_x)
                         .set_input_x2(graph->GetOp(y->Name()));
        graph->AddOp(out->Name(), ge_op);
      }
    } else {
      auto ge_op = ge::op::Add(ge::AscendString(out->Name().c_str()))
                       .set_input_x1(graph->GetOp(x->Name()))
                       .set_input_x2(graph->GetOp(y->Name()));
      graph->AddOp(out->Name(), ge_op);
    }
  }
};

class ElementwiseAddGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto x = ctx.Input("X");
    auto x_dims = x->dims();
    auto y = ctx.Input("Y");
    auto y_dims = y->dims();
    auto out_grad = ctx.Input("Out@GRAD");
    auto out_grad_dim = out_grad->dims();
    auto x_grad = ctx.Output("X@GRAD");
    auto y_grad = ctx.Output("Y@GRAD");

    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(static_cast<int>(x_dims.size()) -
                                  static_cast<int>(y_dims.size()))
                       : axis);

    if (x_grad) {
      if (x_dims == out_grad_dim) {
        graph->RecordNode(x_grad->Name(), graph->GetOp(out_grad->Name()));
      } else {
        std::vector<int64_t> reduce_axes;
        for (auto i = 0; i < x_dims.size(); ++i) {
          if (x_dims[i] == 1 && out_grad_dim[i - axis] > 1) {
            reduce_axes.push_back(i - axis);
          }
        }
        if (reduce_axes.size() > 0) {
          auto ge_op = ge::op::ReduceSumD()
                           .set_input_x(graph->GetOp(out_grad->Name()))
                           .set_attr_axes(reduce_axes)
                           .set_attr_keep_dims(true);
          graph->RecordNode(x_grad->Name() + "_temp", ge_op);
        } else {
          graph->RecordNode(x_grad->Name() + "_temp",
                            graph->GetOp(out_grad->Name()));
        }

        reduce_axes.clear();
        for (auto i = 0; i < out_grad_dim.size(); ++i) {
          if (i < axis || i >= axis + x_dims.size()) {
            reduce_axes.push_back(i);
          }
        }
        if (reduce_axes.size() > 0) {
          auto ge_op = ge::op::ReduceSumD()
                           .set_input_x(graph->GetOp(x_grad->Name() + "_temp"))
                           .set_attr_axes(reduce_axes)
                           .set_attr_keep_dims(false);
          graph->AddOp(x_grad->Name(), ge_op);
        } else {
          graph->AddOp(x_grad->Name(), graph->GetOp(x_grad->Name() + "_temp"));
        }
      }
    }
    if (y_grad) {
      if (y_dims == out_grad_dim) {
        graph->RecordNode(y_grad->Name(), graph->GetOp(out_grad->Name()));
      } else {
        std::vector<int64_t> reduce_axes;
        for (auto i = 0; i < y_dims.size(); ++i) {
          if (y_dims[i] == 1 && out_grad_dim[i - axis] > 1) {
            reduce_axes.push_back(i - axis);
          }
        }
        if (reduce_axes.size() > 0) {
          auto ge_op = ge::op::ReduceSumD()
                           .set_input_x(graph->GetOp(out_grad->Name()))
                           .set_attr_axes(reduce_axes)
                           .set_attr_keep_dims(true);
          graph->RecordNode(y_grad->Name() + "_temp", ge_op);
        } else {
          graph->RecordNode(y_grad->Name() + "_temp",
                            graph->GetOp(out_grad->Name()));
        }

        reduce_axes.clear();
        for (auto i = 0; i < out_grad_dim.size(); ++i) {
          if (i < axis || i >= axis + y_dims.size()) {
            reduce_axes.push_back(i);
          }
        }
        if (reduce_axes.size() > 0) {
          auto ge_op = ge::op::ReduceSumD()
                           .set_input_x(graph->GetOp(y_grad->Name() + "_temp"))
                           .set_attr_axes(reduce_axes)
                           .set_attr_keep_dims(false);
          graph->AddOp(y_grad->Name(), ge_op);
        } else {
          graph->AddOp(y_grad->Name(), graph->GetOp(y_grad->Name() + "_temp"));
        }
      }
    }
  }
};

class ElementwiseMulAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto out = ctx.Output("Out");
    auto x = ctx.Input("X");
    auto y = ctx.Input("Y");
    auto x_dims = x->dims();
    auto y_dims = y->dims();

    auto axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(static_cast<int>(x_dims.size()) -
                                  static_cast<int>(y_dims.size()))
                       : axis);

    if (x_dims.size() >= y_dims.size() &&
        y_dims.size() + axis < x_dims.size()) {
      auto y_dims_tmp = y_dims;
      for (auto i = 0; i < x_dims.size() - y_dims.size() - axis; ++i) {
        y_dims_tmp.push_back(1);
      }
      auto reshape_y =
          graph::funcs::reshape(graph->GetOp(y->Name()), y_dims_tmp);
      auto ge_op = ge::op::Mul(ge::AscendString(out->Name().c_str()))
                       .set_input_x1(graph->GetOp(x->Name()))
                       .set_input_x2(reshape_y);
      graph->AddOp(out->Name(), ge_op);
    } else if (x_dims.size() < y_dims.size() &&
               x_dims.size() + axis < y_dims.size()) {
      auto x_dims_tmp = x_dims;
      for (auto i = 0; i < y_dims.size() - x_dims.size() - axis; ++i) {
        x_dims_tmp.push_back(1);
      }
      auto reshape_x =
          graph::funcs::reshape(graph->GetOp(x->Name()), x_dims_tmp);
      auto ge_op = ge::op::Mul(ge::AscendString(out->Name().c_str()))
                       .set_input_x1(reshape_x)
                       .set_input_x2(graph->GetOp(y->Name()));
      graph->AddOp(out->Name(), ge_op);
    } else {
      auto ge_op = ge::op::Mul(ge::AscendString(out->Name().c_str()))
                       .set_input_x1(graph->GetOp(x->Name()))
                       .set_input_x2(graph->GetOp(y->Name()));
      graph->AddOp(out->Name(), ge_op);
    }
  }
};

class ElementwiseMulGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto x = ctx.Input("X");
    auto x_dims = x->dims();
    auto y = ctx.Input("Y");
    auto y_dims = y->dims();
    auto out_grad = ctx.Input("Out@GRAD");
    auto out_grad_dim = out_grad->dims();
    auto x_grad = ctx.Output("X@GRAD");
    auto y_grad = ctx.Output("Y@GRAD");

    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(static_cast<int>(x_dims.size()) -
                                  static_cast<int>(y_dims.size()))
                       : axis);

    if (x_grad) {
      auto y_dims_tmp = y_dims;
      for (auto i = 0; i < out_grad_dim.size() - y_dims.size() - axis; ++i) {
        y_dims_tmp.push_back(1);
      }
      auto reshape_y =
          graph::funcs::reshape(graph->GetOp(y->Name()), y_dims_tmp);

      auto x_grad_temp = ge::op::Mul()
                             .set_input_x1(graph->GetOp(out_grad->Name()))
                             .set_input_x2(reshape_y);
      if (x_dims == out_grad_dim) {
        graph->AddOp(x_grad->Name(), x_grad_temp);
      } else {
        std::vector<int64_t> reduce_axes;
        for (auto i = 0; i < x_dims.size(); ++i) {
          if (x_dims[i] == 1 && out_grad_dim[i - axis] > 1) {
            reduce_axes.push_back(i - axis);
          }
        }
        if (reduce_axes.size() > 0) {
          auto ge_op = ge::op::ReduceSumD()
                           .set_input_x(x_grad_temp)
                           .set_attr_axes(reduce_axes)
                           .set_attr_keep_dims(true);
          graph->RecordNode(x_grad->Name() + "_temp", ge_op);
        } else {
          graph->RecordNode(x_grad->Name() + "_temp", x_grad_temp);
        }

        reduce_axes.clear();
        for (auto i = 0; i < out_grad_dim.size(); ++i) {
          if (i < axis || i >= axis + x_dims.size()) {
            reduce_axes.push_back(i);
          }
        }
        if (reduce_axes.size() > 0) {
          auto ge_op = ge::op::ReduceSumD()
                           .set_input_x(graph->GetOp(x_grad->Name() + "_temp"))
                           .set_attr_axes(reduce_axes)
                           .set_attr_keep_dims(false);
          graph->AddOp(x_grad->Name(), ge_op);
        } else {
          graph->AddOp(x_grad->Name(), graph->GetOp(x_grad->Name() + "_temp"));
        }
      }
    }
    if (y_grad) {
      auto x_dims_tmp = x_dims;
      for (auto i = 0; i < out_grad_dim.size() - x_dims.size() - axis; ++i) {
        x_dims_tmp.push_back(1);
      }
      auto reshape_x =
          graph::funcs::reshape(graph->GetOp(x->Name()), x_dims_tmp);

      auto y_grad_temp = ge::op::Mul().set_input_x1(reshape_x).set_input_x2(
          graph->GetOp(out_grad->Name()));
      if (y_dims == out_grad_dim) {
        graph->RecordNode(y_grad->Name(), y_grad_temp);
      } else {
        std::vector<int64_t> reduce_axes;
        for (auto i = 0; i < y_dims.size(); ++i) {
          if (y_dims[i] == 1 && out_grad_dim[i - axis] > 1) {
            reduce_axes.push_back(i - axis);
          }
        }
        if (reduce_axes.size() > 0) {
          auto ge_op = ge::op::ReduceSumD()
                           .set_input_x(y_grad_temp)
                           .set_attr_axes(reduce_axes)
                           .set_attr_keep_dims(true);
          graph->RecordNode(y_grad->Name() + "_temp", ge_op);
        } else {
          graph->RecordNode(y_grad->Name() + "_temp", y_grad_temp);
        }

        reduce_axes.clear();
        for (auto i = 0; i < out_grad_dim.size(); ++i) {
          if (i < axis || i >= axis + y_dims.size()) {
            reduce_axes.push_back(i);
          }
        }
        if (reduce_axes.size() > 0) {
          auto ge_op = ge::op::ReduceSumD()
                           .set_input_x(graph->GetOp(y_grad->Name() + "_temp"))
                           .set_attr_axes(reduce_axes)
                           .set_attr_keep_dims(false);
          graph->AddOp(y_grad->Name(), ge_op);
        } else {
          graph->AddOp(y_grad->Name(), graph->GetOp(y_grad->Name() + "_temp"));
        }
      }
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(elementwise_add, custom_graph::ElementwiseAddAdapter);
REG_OP_ADAPTER(elementwise_add_grad, custom_graph::ElementwiseAddGradAdapter);
REG_OP_ADAPTER(elementwise_mul, custom_graph::ElementwiseMulAdapter);
REG_OP_ADAPTER(elementwise_mul_grad, custom_graph::ElementwiseMulGradAdapter);
