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

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto out = ctx.Output("Out");
    auto x = ctx.Input("X");
    int start_axis = ctx.Attr<int>("start_axis");
    int stop_axis = ctx.Attr<int>("stop_axis");

    auto ge_op = ge::op::FlattenV2()
                     .set_input_x(graph->GetOp(x->Name()))
                     .set_attr_axis(start_axis)
                     .set_attr_end_axis(stop_axis);
    graph->AddOp(out->Name(), ge_op);
  }
};

class FlattenContiguousRangeGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto x_grad = ctx.Output("X@GRAD");

    auto out_grad = ctx.Input("Out@GRAD");
    auto xshape_dims = ctx.Input("XShape")->dims();
    auto x_dims = slice_ddim(xshape_dims, 1, xshape_dims.size());
    graph::utils::log() << "[INFO] x_grad="
                        << paddle::framework::ir::to_string(x_grad->dims())
                        << std::endl;
    graph::utils::log() << "[INFO] out_grad="
                        << paddle::framework::ir::to_string(out_grad->dims())
                        << std::endl;
    graph::utils::log() << "[INFO] x_dims="
                        << paddle::framework::ir::to_string(x_dims)
                        << std::endl;

    // ge::TensorDesc shape_tensor_desc(
    //     ge::Shape(std::vector<int64_t>({x_dims.size()})),
    //     ge::Format::FORMAT_NCHW,
    //     ge::DataType::DT_INT32);
    // shape_tensor_desc.SetRealDimCnt(shape_tensor_desc.GetShape().GetDimNum());
    // ge::Tensor shape_tensor(shape_tensor_desc,
    //                         reinterpret_cast<uint8_t*>(x_dims.data()),
    //                         x_dims.size() * sizeof(int));

    // auto constant_op = ge::op::Constant().set_attr_value(shape_tensor);
    // constant_op.update_output_desc_y(shape_tensor_desc);

    // auto constant_op = graph::funcs::constant({x_dims.size()}, x_dims);

    // auto ge_op = ge::op::Reshape()
    //                  .set_input_x(graph->GetOp(out_grad->Name()))
    //                  .set_input_shape(constant_op);

    auto ge_op = graph::funcs::reshape(graph->GetOp(out_grad->Name()), x_dims);
    graph->AddOp(x_grad->Name(), ge_op);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(flatten_contiguous_range,
               custom_graph::FlattenContiguousRangeAdapter);
REG_OP_ADAPTER(flatten_contiguous_range_grad,
               custom_graph::FlattenContiguousRangeGradAdapter);
