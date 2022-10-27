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

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& y = ctx.Input("Y");
    auto& out = ctx.Output("Out");
    auto axis = ctx.Attr<int>("axis");

    auto x_dims = x.Shape();
    auto y_dims = y.Shape();
    axis = (axis == -1 ? std::abs(static_cast<int>(x_dims.size()) -
                                  static_cast<int>(y_dims.size()))
                       : axis);

    if (x_dims.size() >= y_dims.size() &&
        y_dims.size() + axis < x_dims.size()) {
      if (x_dims.size() == 4 && y_dims.size() == 1 && axis == 3) {
        OpCommand("BiasAdd").Input(x).Input(y).Output(out).Attr("data_format",
                                                                "NHWC");
      } else if (x_dims.size() == 4 && y_dims.size() == 1 && axis == 1) {
        OpCommand("BiasAdd").Input(x).Input(y).Output(out).Attr("data_format",
                                                                "NCHW");
      } else {
        auto y_dims_tmp = y_dims;
        for (auto i = 0; i < x_dims.size() - y_dims.size() - axis; ++i) {
          y_dims_tmp.push_back(1);
        }
        Tensor reshape_y;
        OpCommand::Reshape(y, reshape_y, y_dims_tmp);
        OpCommand("Add").Input(x).Input(reshape_y).Output(out);
      }
    } else if (x_dims.size() < y_dims.size() &&
               x_dims.size() + axis < y_dims.size()) {
      if (y_dims.size() == 4 && x_dims.size() == 1 && axis == 3) {
        OpCommand("BiasAdd").Input(y).Input(x).Output(out).Attr("data_format",
                                                                "NHWC");
      } else if (y_dims.size() == 4 && x_dims.size() == 1 && axis == 1) {
        OpCommand("BiasAdd").Input(y).Input(x).Output(out).Attr("data_format",
                                                                "NCHW");
      } else {
        auto x_dims_tmp = x_dims;
        for (auto i = 0; i < y_dims.size() - x_dims.size() - axis; ++i) {
          x_dims_tmp.push_back(1);
        }
        Tensor reshape_x;
        OpCommand::Reshape(x, reshape_x, x_dims_tmp);
        OpCommand("Add").Input(reshape_x).Input(y).Output(out);
      }
    } else {
      OpCommand("Add").Input(x).Input(y).Output(out);
    }
  }
};

class ElementwiseAddGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& y = ctx.Input("Y");
    auto& out_grad = ctx.Input("Out@GRAD");
    int axis = ctx.Attr<int>("axis");

    auto x_dims = x.Shape();
    auto y_dims = y.Shape();
    auto out_grad_dim = out_grad.Shape();

    axis = (axis == -1 ? std::abs(static_cast<int>(x_dims.size()) -
                                  static_cast<int>(y_dims.size()))
                       : axis);

    if (ctx.HasOutput("X@GRAD")) {
      auto& x_grad = ctx.Output("X@GRAD");

      if (x_dims == out_grad_dim) {
        x_grad = out_grad;
      } else {
        std::vector<int64_t> reduce_axes;
        for (auto i = 0; i < x_dims.size(); ++i) {
          if (x_dims[i] == 1 && out_grad_dim[i - axis] > 1) {
            reduce_axes.push_back(i - axis);
          }
        }

        Tensor tmp;
        if (reduce_axes.size() > 0) {
          OpCommand("ReduceSumD")
              .Input(out_grad)
              .Output(tmp)
              .Attr("axes", reduce_axes)
              .Attr("keep_dims", true);
        } else {
          tmp = out_grad;
        }
        reduce_axes.clear();
        for (auto i = 0; i < out_grad_dim.size(); ++i) {
          if (i < axis || i >= axis + x_dims.size()) {
            reduce_axes.push_back(i);
          }
        }
        if (reduce_axes.size() > 0) {
          OpCommand("ReduceSumD")
              .Input(tmp)
              .Output(x_grad)
              .Attr("axes", reduce_axes)
              .Attr("keep_dims", false);
        }
      }
    }
    if (ctx.HasOutput("Y@GRAD")) {
      auto& y_grad = ctx.Output("Y@GRAD");

      if (y_dims == out_grad_dim) {
        y_grad = out_grad;
      } else {
        std::vector<int64_t> reduce_axes;
        for (auto i = 0; i < y_dims.size(); ++i) {
          if (y_dims[i] == 1 && out_grad_dim[i - axis] > 1) {
            reduce_axes.push_back(i - axis);
          }
        }

        Tensor tmp;
        if (reduce_axes.size() > 0) {
          OpCommand("ReduceSumD")
              .Input(out_grad)
              .Output(tmp)
              .Attr("axes", reduce_axes)
              .Attr("keep_dims", true);
        } else {
          tmp = out_grad;
        }
        reduce_axes.clear();
        for (auto i = 0; i < out_grad_dim.size(); ++i) {
          if (i < axis || i >= axis + y_dims.size()) {
            reduce_axes.push_back(i);
          }
        }
        if (reduce_axes.size() > 0) {
          OpCommand("ReduceSumD")
              .Input(tmp)
              .Output(y_grad)
              .Attr("axes", reduce_axes)
              .Attr("keep_dims", false);
        }
      }
    }
  }
};

class ElementwiseMulAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& y = ctx.Input("Y");
    auto& out = ctx.Output("Out");

    auto x_dims = x.Shape();
    auto y_dims = y.Shape();

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
      Tensor reshape_y;
      OpCommand::Reshape(y, reshape_y, y_dims_tmp);
      OpCommand("Mul").Input(x).Input(reshape_y).Output(out);
    } else if (x_dims.size() < y_dims.size() &&
               x_dims.size() + axis < y_dims.size()) {
      auto x_dims_tmp = x_dims;
      for (auto i = 0; i < y_dims.size() - x_dims.size() - axis; ++i) {
        x_dims_tmp.push_back(1);
      }
      Tensor reshape_x;
      OpCommand::Reshape(reshape_x, x, x_dims_tmp);
      OpCommand("Mul").Input(reshape_x).Input(y).Output(out);
    } else {
      OpCommand("Mul").Input(x).Input(y).Output(out);
    }
  }
};

class ElementwiseMulGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& y = ctx.Input("Y");
    auto& out_grad = ctx.Input("Out@GRAD");
    auto x_dims = x.Shape();
    auto y_dims = y.Shape();
    auto out_grad_dim = out_grad.Shape();

    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(static_cast<int>(x_dims.size()) -
                                  static_cast<int>(y_dims.size()))
                       : axis);

    if (ctx.HasOutput("X@GRAD")) {
      auto& x_grad = ctx.Output("X@GRAD");
      auto y_dims_tmp = y_dims;
      for (auto i = 0; i < out_grad_dim.size() - y_dims.size() - axis; ++i) {
        y_dims_tmp.push_back(1);
      }
      Tensor reshape_y, x_grad_tmp;
      OpCommand::Reshape(y, reshape_y, y_dims_tmp);
      OpCommand("Mul").Input(out_grad).Input(reshape_y).Output(x_grad_tmp);

      if (x_dims == out_grad_dim) {
        x_grad = x_grad_tmp;
      } else {
        std::vector<int64_t> reduce_axes;
        for (auto i = 0; i < x_dims.size(); ++i) {
          if (x_dims[i] == 1 && out_grad_dim[i - axis] > 1) {
            reduce_axes.push_back(i - axis);
          }
        }

        Tensor x_grad_tmp_2;
        if (reduce_axes.size() > 0) {
          OpCommand("ReduceSumD")
              .Input(x_grad_tmp)
              .Output(x_grad_tmp_2)
              .Attr("axes", reduce_axes)
              .Attr("keep_dims", true);
        } else {
          x_grad_tmp_2 = x_grad_tmp;
        }

        reduce_axes.clear();
        for (auto i = 0; i < out_grad_dim.size(); ++i) {
          if (i < axis || i >= axis + x_dims.size()) {
            reduce_axes.push_back(i);
          }
        }
        if (reduce_axes.size() > 0) {
          OpCommand("ReduceSumD")
              .Input(x_grad_tmp_2)
              .Output(x_grad)
              .Attr("axes", reduce_axes)
              .Attr("keep_dims", false);
        } else {
          x_grad = x_grad_tmp_2;
        }
      }
    }
    if (ctx.HasOutput("Y@GRAD")) {
      auto& y_grad = ctx.Output("Y@GRAD");
      auto x_dims_tmp = x_dims;
      for (auto i = 0; i < out_grad_dim.size() - x_dims.size() - axis; ++i) {
        x_dims_tmp.push_back(1);
      }

      Tensor reshape_x, y_grad_tmp;
      OpCommand::Reshape(x, reshape_x, x_dims_tmp);
      OpCommand("Mul").Input(reshape_x).Input(out_grad).Output(y_grad_tmp);

      if (y_dims == out_grad_dim) {
        y_grad = y_grad_tmp;
      } else {
        std::vector<int64_t> reduce_axes;
        for (auto i = 0; i < y_dims.size(); ++i) {
          if (y_dims[i] == 1 && out_grad_dim[i - axis] > 1) {
            reduce_axes.push_back(i - axis);
          }
        }

        Tensor y_grad_tmp_2;
        if (reduce_axes.size() > 0) {
          OpCommand("ReduceSumD")
              .Input(y_grad_tmp)
              .Output(y_grad_tmp_2)
              .Attr("axes", reduce_axes)
              .Attr("keep_dims", true);
        } else {
          y_grad_tmp_2 = y_grad_tmp;
        }

        reduce_axes.clear();
        for (auto i = 0; i < out_grad_dim.size(); ++i) {
          if (i < axis || i >= axis + y_dims.size()) {
            reduce_axes.push_back(i);
          }
        }
        if (reduce_axes.size() > 0) {
          OpCommand("ReduceSumD")
              .Input(y_grad_tmp_2)
              .Output(y_grad)
              .Attr("axes", reduce_axes)
              .Attr("keep_dims", false);
        } else {
          y_grad = y_grad_tmp_2;
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
