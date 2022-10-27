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

class Conv2dAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& input = ctx.Input("Input");
    auto& filter = ctx.Input("Filter");
    auto& output = ctx.Output("Output");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    std::string data_format = ctx.Attr<std::string>("data_format");
    const bool channel_last = data_format == "NHWC";

    auto in_dims = input.Shape();
    auto filter_dims = filter.Shape();
    std::vector<int> in_data_dims;

    if (channel_last) {
      in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
    }
    std::vector<int> ksize = slice_ddim(filter_dims, 2, in_dims.size());

    UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

    std::vector<int64_t> strides_vec(4, 1);
    std::vector<int64_t> dilations_vec(4, 1);
    if (channel_last) {
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
      dilations_vec[1] = dilations[0];
      dilations_vec[2] = dilations[1];
    } else {
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
      dilations_vec[2] = dilations[0];
      dilations_vec[3] = dilations[1];
    }

    OpCommand("Conv2D")
        .Input(input, "x")
        .Input(filter, "filter")
        .Output(output, "y")
        .Attr("strides", strides_vec)
        .Attr("pads", std::vector<int64_t>(paddings.begin(), paddings.end()))
        .Attr("dilations", dilations_vec)
        .Attr("groups", groups)
        .Attr("data_format", data_format);
  }
};

class Conv2dGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& input = ctx.Input("Input");
    auto& filter = ctx.Input("Filter");
    auto& output_grad = ctx.Input("Output@GRAD");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    std::string data_format = ctx.Attr<std::string>("data_format");
    const bool channel_last = data_format == "NHWC";

    auto in_dims = input.Shape();
    auto filter_dims = filter.Shape();
    std::vector<int> in_data_dims;

    if (channel_last) {
      in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
    }
    std::vector<int> ksize = slice_ddim(filter_dims, 2, in_dims.size());

    UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

    std::vector<int64_t> strides_vec(4, 1);
    std::vector<int64_t> dilations_vec(4, 1);
    if (channel_last) {
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
      dilations_vec[1] = dilations[0];
      dilations_vec[2] = dilations[1];
    } else {
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
      dilations_vec[2] = dilations[0];
      dilations_vec[3] = dilations[1];
    }

    if (ctx.HasOutput("Input@GRAD")) {
      auto& input_grad = ctx.Output("Input@GRAD");

      Tensor input_size;
      OpCommand::FillConstant(
          input_size,
          {in_dims.size()},
          std::vector<int64_t>(in_dims.begin(), in_dims.end()));
      OpCommand("Conv2DBackpropInputD")
          .Input(filter, "filter")
          .Input(output_grad, "out_backprop")
          .Output(input_grad, "y")
          .Attr("input_size",
                std::vector<int64_t>(in_dims.begin(), in_dims.end()))
          .Attr("strides", strides_vec)
          .Attr("pads", std::vector<int64_t>(paddings.begin(), paddings.end()))
          .Attr("dilations", dilations_vec)
          .Attr("groups", groups)
          .Attr("data_format", data_format);
    }

    if (ctx.HasOutput("Filter@GRAD")) {
      auto& filter_grad = ctx.Output("Filter@GRAD");

      Tensor filter_size;
      OpCommand::FillConstant(
          filter_size,
          {filter_dims.size()},
          std::vector<int64_t>(filter_dims.begin(), filter_dims.end()));
      OpCommand("Conv2DBackpropFilterD")
          .Input(input, "x")
          .Input(output_grad, "out_backprop")
          .Output(filter_grad, "y")
          .Attr("filter_size",
                std::vector<int64_t>(filter_dims.begin(), filter_dims.end()))
          .Attr("strides", strides_vec)
          .Attr("pads", std::vector<int64_t>(paddings.begin(), paddings.end()))
          .Attr("dilations", dilations_vec)
          .Attr("groups", groups)
          .Attr("data_format", data_format);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(conv2d, custom_graph::Conv2dAdapter);
REG_OP_ADAPTER(conv2d_grad, custom_graph::Conv2dGradAdapter);
