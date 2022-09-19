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

class FillConstantAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto out = ctx.Output("Out");
    auto out_dims = out->dims();
    auto fill_constant_value = ctx.Attr<float>("value");
    std::cout << "fill_constant_value=" << fill_constant_value << std::endl;

    ge::TensorDesc out_tensor_desc(
        ge::Shape(std::vector<int64_t>(out_dims.begin(), out_dims.end())),
        ge::Format::FORMAT_NCHW,
        graph::utils::pd_dtype_to_ge_dtype(out->dtype()));
    out_tensor_desc.SetRealDimCnt(out_tensor_desc.GetShape().GetDimNum());

    auto size = std::accumulate(
        out_dims.begin(), out_dims.end(), 1, std::multiplies<int>());
    auto bytesize = size;

    uint8_t* data_value = nullptr;
    if (out->dtype() == paddle::framework::proto::VarType::FP32) {
      auto ptr = new float[size];
      for (auto i = 0; i < size; ++i) {
        ptr[i] = static_cast<float>(fill_constant_value);
      }
      bytesize = size * sizeof(float);
      data_value = reinterpret_cast<uint8_t*>(ptr);
    } else if (out->dtype() == paddle::framework::proto::VarType::FP64) {
      auto ptr = new double[size];
      for (auto i = 0; i < size; ++i) {
        ptr[i] = static_cast<double>(fill_constant_value);
      }
      bytesize = size * sizeof(double);
      data_value = reinterpret_cast<uint8_t*>(ptr);
    } else {
      std::cerr << "fill_constant unsupported datatype " << out->dtype();
      exit(-1);
    }

    ge::Tensor tensor(out_tensor_desc, data_value, bytesize);

    auto constant_op = ge::op::Constant().set_attr_value(tensor);
    constant_op.update_output_desc_y(out_tensor_desc);

    // for loss grad
    if (!graph->HasOp(out->Name())) {
      std::string var_name = out->Name();
      auto var_dims = out->dims();
      ge::TensorDesc var_desc(
          ge::Shape(std::vector<int64_t>(var_dims.begin(), var_dims.end())),
          ge::Format::FORMAT_NCHW,
          graph::utils::pd_dtype_to_ge_dtype(out->dtype()));
      var_desc.SetRealDimCnt(var_desc.GetShape().GetDimNum());

      auto ge_op = ge::op::Variable();
      ge_op.update_output_desc_y(var_desc);
      graph->AddOp(var_name, ge_op);
    }

    // inplace op
    auto assign_op = ge::op::Assign()
                         .set_input_ref(graph->GetOp(out->Name()))
                         .set_input_value(constant_op);
    graph->AddInput(graph->GetOp(out->Name()));
    // graph->Graph()->AddOp(assign_op);

    std::cout << "fill constant tensor: " << out->Name()
              << ", dims: " << paddle::framework::ir::to_string(out->dims())
              << std::endl;

    if (out->dtype() == paddle::framework::proto::VarType::FP32) {
      auto ptr = reinterpret_cast<float*>(data_value);
      delete[] ptr;
    } else if (out->dtype() == paddle::framework::proto::VarType::FP64) {
      auto ptr = reinterpret_cast<double*>(data_value);
      delete[] ptr;
    } else {
      std::cerr << "fill_constant unsupported datatype " << out->dtype();
      exit(-1);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(fill_constant, custom_graph::FillConstantAdapter);
