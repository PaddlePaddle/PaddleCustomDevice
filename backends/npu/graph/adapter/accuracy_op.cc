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

class AccuracyAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& inference = ctx.Input("Out");
    auto& label = ctx.Input("Label");
    auto& indices = ctx.Input("Indices");
    auto& accuracy = ctx.Output("Accuracy");
    auto& correct = ctx.Output("Correct");
    auto& total = ctx.Output("Total");

    int num_samples = inference.Shape()[0];
    if (num_samples == 0) {
      return;
    }

    Tensor label_same_dtype_with_indices;

    if (label.DType() != indices.DType()) {
      OpCommand("Cast")
          .Input(label)
          .Output(label_same_dtype_with_indices)
          .Attr("dst_type",
                static_cast<int>(
                    graph::utils::pd_dtype_to_ge_dtype(indices.DType())));
    } else {
      label_same_dtype_with_indices = label;
    }

    OpCommandPipe()
        .Op("Equal")
        .Op("Cast")
        .Attr("dst_type",
              static_cast<int>(
                  graph::utils::cpp_type_to_ge_dtype<float>::value()))
        .Op("ReduceMaxD")
        .Attr("axes", std::vector<int64_t>({1}))
        .Attr("keep_dims", false)
        .Op("ReduceSumD")
        .Attr("axes", std::vector<int64_t>({0}))
        .Attr("keep_dims", false)
        .Input(indices)
        .Input(label_same_dtype_with_indices)
        .Output(correct)
        .End();

    OpCommand::FillConstant(
        total,
        {1},
        std::vector<float>({static_cast<float>(num_samples)}),
        ge::Format::FORMAT_NCHW);
    OpCommand("Div").Input(correct).Input(total).Output(accuracy);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(accuracy, custom_graph::AccuracyAdapter);
