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

class Pool2dAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& in_x = ctx.Input("X");
    auto& out = ctx.Output("Out");

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::string data_format = ctx.Attr<std::string>("data_format");

    bool global_pooling = ctx.Attr<bool>("global_pooling");
    bool ceil_mode = ctx.Attr<bool>("ceil_mode");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool adaptive = ctx.Attr<bool>("adaptive");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");

    const bool channel_last = data_format == "NHWC";

    auto in_x_dims = in_x.Shape();
    auto out_dims = out.Shape();

    std::vector<int64_t> ksize_vec(4, 1);
    std::vector<int64_t> strides_vec(4, 1);

    std::vector<int> data_dims;
    std::vector<int> out_data_dims;

    if (channel_last) {
      data_dims = slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
      out_data_dims = slice_ddim(out_dims, 1, out_dims.size() - 1);
      ksize_vec[1] = ksize[0];
      ksize_vec[2] = ksize[1];
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
    } else {
      data_dims = slice_ddim(in_x_dims, 2, in_x_dims.size());
      out_data_dims = slice_ddim(out_dims, 2, out_dims.size());
      ksize_vec[2] = ksize[0];
      ksize_vec[3] = ksize[1];
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
    }
    UpdatePadding(&paddings,
                  global_pooling,
                  adaptive,
                  padding_algorithm,
                  data_dims,
                  strides,
                  ksize);

    if (adaptive) {
      // AdaptiveAvgPool2d only support NCHW
      if (pooling_type == "avg") {
        if (channel_last) {
          Tensor trans_x, trans_out;

          OpCommandPipe()
              .Op("TransData")
              .Attr("src_format", "NHWC")
              .Attr("dst_format", "NCHW")
              .Op("AdaptiveAvgPool2d")
              .Attr("output_size",
                    std::vector<int64_t>(out_data_dims.begin(),
                                         out_data_dims.end()))
              .Op("TransData")
              .Attr("src_format", "NCHW")
              .Attr("dst_format", "NHWC")
              .Input(in_x)
              .Output(out)
              .End();
        } else {
          OpCommand("AdaptiveAvgPool2d")
              .Input(in_x)
              .Output(out)
              .Attr("output_size",
                    std::vector<int64_t>(out_data_dims.begin(),
                                         out_data_dims.end()));
        }
      } else {
        OpCommand("AdaptiveMaxPool2d")
            .Input(in_x)
            .Output(out)
            .Attr("output_size",
                  std::vector<int64_t>(out_data_dims.begin(),
                                       out_data_dims.end()));
      }
    } else {
      if (pooling_type == "max") {
        OpCommand("MaxPoolV3")
            .Input(in_x, "x")
            .Output(out, "y")
            .Attr("ksize", ksize_vec)
            .Attr("strides", strides_vec)
            .Attr("padding_mode", "CALCULATED")
            .Attr("pads",
                  std::vector<int64_t>(paddings.begin(), paddings.end()))
            .Attr("data_format", data_format)
            .Attr("global_pooling", global_pooling)
            .Attr("ceil_mode", ceil_mode);
      } else {
        OpCommand("AvgPoolV2")
            .Input(in_x, "x")
            .Output(out, "y")
            .Attr("ksize", ksize_vec)
            .Attr("strides", strides_vec)
            .Attr("padding_mode", "CALCULATED")
            .Attr("pads",
                  std::vector<int64_t>(paddings.begin(), paddings.end()))
            .Attr("data_format", data_format)
            .Attr("global_pooling", global_pooling)
            .Attr("ceil_mode", ceil_mode)
            .Attr("exclusive", exclusive);
      }
    }
  }
};

class Pool2dGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& in_x = ctx.Input("X");
    auto& out = ctx.Input("Out");
    auto& out_grad = ctx.Input(paddle::framework::GradVarName("Out"));
    auto& in_x_grad = ctx.Output(paddle::framework::GradVarName("X"));

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    bool ceil_mode = ctx.Attr<bool>("ceil_mode");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool adaptive = ctx.Attr<bool>("adaptive");
    std::string data_format = ctx.Attr<std::string>("data_format");
    bool global_pooling = ctx.Attr<bool>("global_pooling");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");

    const bool channel_last = data_format == "NHWC";

    // update paddings
    auto in_x_dims = in_x.Shape();
    auto out_dims = out.Shape();

    std::vector<int64_t> ksize_vec(4, 1);
    std::vector<int64_t> strides_vec(4, 1);

    std::vector<int> data_dims;
    std::vector<int> out_data_dims;

    if (channel_last) {
      data_dims = slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
      out_data_dims = slice_ddim(out_dims, 1, out_dims.size() - 1);
      ksize_vec[1] = ksize[0];
      ksize_vec[2] = ksize[1];
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
    } else {
      data_dims = slice_ddim(in_x_dims, 2, in_x_dims.size());
      out_data_dims = slice_ddim(out_dims, 2, out_dims.size());
      ksize_vec[2] = ksize[0];
      ksize_vec[3] = ksize[1];
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
    }
    UpdatePadding(&paddings,
                  global_pooling,
                  adaptive,
                  padding_algorithm,
                  data_dims,
                  strides,
                  ksize);

    if (adaptive || (global_pooling && pooling_type == "max")) {
      if (channel_last) {
        strides_vec[1] = data_dims[0] / out_data_dims[0];
        strides_vec[2] = data_dims[1] / out_data_dims[1];
        ksize_vec[1] = strides_vec[1];
        ksize_vec[2] = strides_vec[2];
      } else {
        strides_vec[2] = data_dims[0] / out_data_dims[0];
        strides_vec[3] = data_dims[1] / out_data_dims[1];
        ksize_vec[2] = strides_vec[2];
        ksize_vec[3] = strides_vec[3];
      }
    }

    if (pooling_type == "max") {
      if (global_pooling) {
        for (auto& s : strides_vec) {
          s = 1;
        }
        global_pooling = false;
      }

      OpCommand("MaxPoolV3Grad")
          .Input(in_x, "orig_input")
          .Input(out, "orig_output")
          .Input(out_grad, "grad")
          .Output(in_x_grad, "out_grad")
          .Attr("ksize", ksize_vec)
          .Attr("strides", strides_vec)
          .Attr("padding_mode", "CALCULATED")
          .Attr("pads", std::vector<int64_t>(paddings.begin(), paddings.end()))
          .Attr("data_format", data_format)
          .Attr("global_pooling", global_pooling)
          .Attr("ceil_mode", ceil_mode);
    } else if (pooling_type == "avg") {
      Tensor input_shape;
      OpCommand::FillConstant(input_shape,
                              {static_cast<int>(in_x_dims.size())},
                              std::move(in_x_dims));
      OpCommand("AvgPoolV2Grad")
          .Input(input_shape)
          .Input(out_grad, "input_grad")
          .Output(in_x_grad, "out_grad")
          .Attr("ksize", ksize_vec)
          .Attr("strides", strides_vec)
          .Attr("padding_mode", "CALCULATED")
          .Attr("pads", std::vector<int64_t>(paddings.begin(), paddings.end()))
          .Attr("data_format", data_format)
          .Attr("global_pooling", global_pooling)
          .Attr("ceil_mode", ceil_mode)
          .Attr("exclusive", exclusive);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(pool2d, custom_graph::Pool2dAdapter);
REG_OP_ADAPTER(pool2d_grad, custom_graph::Pool2dGradAdapter);
