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

  void run(const paddle::framework::ir::OpNode &ctx,
           custom_graph::GEGraph *graph) override {
    auto in_x = ctx.Input("X");
    auto out = ctx.Output("Out");

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

    auto in_x_dims = in_x->dims();
    auto out_dims = out->dims();

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
      graph::utils::log() << "[ERROR] unsupport adaptive pooling\n";
      exit(-1);
    } else {
      if (pooling_type == "max") {
        // PADDLE_ENFORCE_EQ(
        //     exclusive,
        //     true,
        //     platform::errors::InvalidArgument(
        //         "MaxPool only support exclusive=false, but got true"));
        auto ge_op = ge::op::MaxPoolV3()
                         .set_input_x(graph->GetOp(in_x->Name()))
                         .set_attr_ksize(ksize_vec)
                         .set_attr_strides(strides_vec)
                         .set_attr_padding_mode(std::string("CALCULATED"))
                         .set_attr_pads(std::vector<int64_t>(paddings.begin(),
                                                             paddings.end()))
                         .set_attr_data_format(data_format)
                         .set_attr_global_pooling(global_pooling)
                         .set_attr_ceil_mode(ceil_mode);
        graph->AddOp(out->Name(), ge_op);
      } else {
        auto ge_op = ge::op::AvgPoolV2()
                         .set_input_x(graph->GetOp(in_x->Name()))
                         .set_attr_ksize(ksize_vec)
                         .set_attr_strides(strides_vec)
                         .set_attr_padding_mode(std::string("CALCULATED"))
                         .set_attr_pads(std::vector<int64_t>(paddings.begin(),
                                                             paddings.end()))
                         .set_attr_data_format(data_format)
                         .set_attr_global_pooling(global_pooling)
                         .set_attr_ceil_mode(ceil_mode)
                         .set_attr_exclusive(exclusive);
        graph->AddOp(out->Name(), ge_op);
      }
    }
  }
};

class Pool2dGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode &ctx,
           custom_graph::GEGraph *graph) override {
    auto *in_x = ctx.Input("X");
    auto *out = ctx.Input("Out");
    auto *out_grad = ctx.Input(paddle::framework::GradVarName("Out"));
    auto *in_x_grad = ctx.Output(paddle::framework::GradVarName("X"));

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
    auto in_x_dims = in_x->dims();
    auto out_dims = out->dims();

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
      // PADDLE_ENFORCE_EQ(data_dims[0] % out_data_dims[0],
      //                   0,
      //                   platform::errors::InvalidArgument(
      //                       "When adaptive = True, H and W must be divisible,
      //                       " "but input dims is %s, output dims is %s",
      //                       data_dims,
      //                       out_data_dims));
      // PADDLE_ENFORCE_EQ(data_dims[1] % out_data_dims[1],
      //                   0,
      //                   platform::errors::InvalidArgument(
      //                       "When adaptive = True, H and W must be divisible,
      //                       " "but input dims is %s, output dims is %s",
      //                       data_dims,
      //                       out_data_dims));
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
        for (auto &s : strides_vec) {
          s = 1;
        }
        // PADDLE_ENFORCE_LT(std::max(data_dims[0], data_dims[1]),
        //                   255,
        //                   platform::errors::InvalidArgument(
        //                       "MaxPoolGrad H, W must be less than 255 when "
        //                       "global_pooling = True, but got %s",
        //                       data_dims));
        global_pooling = false;
      }

      auto ge_op = ge::op::MaxPoolV3Grad()
                       .set_input_orig_input(graph->GetOp(in_x->Name()))
                       .set_input_orig_output(graph->GetOp(out->Name()))
                       .set_input_grad(graph->GetOp(out_grad->Name()))
                       .set_attr_ksize(ksize_vec)
                       .set_attr_strides(strides_vec)
                       .set_attr_padding_mode("CALCULATED")
                       .set_attr_pads(std::vector<int64_t>(paddings.begin(),
                                                           paddings.end()))
                       .set_attr_data_format(data_format)
                       .set_attr_global_pooling(global_pooling)
                       .set_attr_ceil_mode(ceil_mode);  // 0: floor, 1: ceil

      graph->AddOp(in_x_grad->Name(), ge_op);
    } else if (pooling_type == "avg") {
      // PADDLE_ENFORCE(strides[0] == strides[1],
      //                platform::errors::InvalidArgument(
      //                    "AvgPoolGrad dose not support Asymmetric strides.
      //                    but " "strides = (%d, %d)", strides[0],
      //                    strides[1]));

      auto input_shape = graph::funcs::constant(
          {static_cast<int>(in_x_dims.size())}, std::move(in_x_dims));

      auto ge_op = ge::op::AvgPoolV2Grad()
                       .set_input_orig_input_shape(input_shape)
                       .set_input_input_grad(graph->GetOp(out_grad->Name()))
                       .set_attr_ksize(ksize_vec)
                       .set_attr_strides(strides_vec)
                       .set_attr_padding_mode("CALCULATED")
                       .set_attr_pads(std::vector<int64_t>(paddings.begin(),
                                                           paddings.end()))
                       .set_attr_data_format(data_format)
                       .set_attr_global_pooling(global_pooling)
                       .set_attr_ceil_mode(ceil_mode)  // 0: floor, 1: ceil
                       .set_attr_exclusive(exclusive);

      graph->AddOp(in_x_grad->Name(), ge_op);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(pool2d, custom_graph::Pool2dAdapter);
REG_OP_ADAPTER(pool2d_grad, custom_graph::Pool2dGradAdapter);
