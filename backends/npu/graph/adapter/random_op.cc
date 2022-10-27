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

class UniformRandomAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto out = ctx.Output("Out");
    auto out_dims = out.Shape();
    auto min_value = ctx.Attr<float>("min");
    auto max_value = ctx.Attr<float>("max");
    auto seed = ctx.Attr<int>("seed");
    // graph::utils::log() << "[INFO] min_value=" << min_value << std::endl;
    // graph::utils::log() << "[INFO] max_value=" << max_value << std::endl;

    ge::TensorDesc out_tensor_desc(
        ge::Shape(std::vector<int64_t>(out_dims.begin(), out_dims.end())),
        ge::Format::FORMAT_NCHW,
        graph::utils::pd_dtype_to_ge_dtype(out.DType()));
    out_tensor_desc.SetRealDimCnt(out_tensor_desc.GetShape().GetDimNum());

    auto size = std::accumulate(
        out_dims.begin(), out_dims.end(), 1, std::multiplies<int>());
    auto bytesize = size;

    auto engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);

    Tensor constant_tensor;
    if (out.DType() == paddle::framework::proto::VarType::FP16) {
      std::uniform_real_distribution<float> dist(static_cast<float>(min_value),
                                                 static_cast<float>(max_value));
      auto value =
          std::shared_ptr<phi::dtype::float16>(new phi::dtype::float16[size]);
      for (auto i = 0; i < size; ++i) {
        *(value.get() + i) = static_cast<phi::dtype::float16>(dist(*engine));
      }
      OpCommand::FillConstant(constant_tensor, out_dims, value.get());
    } else if (out.DType() == paddle::framework::proto::VarType::FP32) {
      std::uniform_real_distribution<float> dist(static_cast<float>(min_value),
                                                 static_cast<float>(max_value));
      auto value = std::shared_ptr<float>(new float[size]);
      for (auto i = 0; i < size; ++i) {
        *(value.get() + i) = static_cast<float>(dist(*engine));
      }
      OpCommand::FillConstant(constant_tensor, out_dims, value.get());
    } else if (out.DType() == paddle::framework::proto::VarType::FP64) {
      std::uniform_real_distribution<double> dist(
          static_cast<double>(min_value), static_cast<double>(max_value));
      auto value = std::shared_ptr<double>(new double[size]);
      for (auto i = 0; i < size; ++i) {
        *(value.get() + i) = static_cast<double>(dist(*engine));
      }
      OpCommand::FillConstant(constant_tensor, out_dims, value.get());
    } else {
      graph::utils::log() << "[ERROR] fill_constant unsupported datatype "
                          << out.DType() << std::endl;
      exit(-1);
    }

    if (out.op() == nullptr) {
      out = constant_tensor;
    } else {
      OpCommand("Assign").Input(out).Input(constant_tensor);
    }
  }
};

class GaussianRandomAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto out = ctx.Output("Out");
    auto out_dims = out.Shape();
    auto mean_value = ctx.Attr<float>("mean");
    auto std_value = ctx.Attr<float>("std");
    auto seed = ctx.Attr<int>("seed");
    graph::utils::log() << "[INFO] mean_value=" << mean_value << std::endl;
    graph::utils::log() << "[INFO] std_value=" << std_value << std::endl;

    ge::TensorDesc out_tensor_desc(
        ge::Shape(std::vector<int64_t>(out_dims.begin(), out_dims.end())),
        ge::Format::FORMAT_NCHW,
        graph::utils::pd_dtype_to_ge_dtype(out.DType()));
    out_tensor_desc.SetRealDimCnt(out_tensor_desc.GetShape().GetDimNum());

    auto size = std::accumulate(
        out_dims.begin(), out_dims.end(), 1, std::multiplies<int>());
    auto bytesize = size;

    uint8_t* data_value = nullptr;

    auto engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);

    Tensor constant_tensor;
    if (out.DType() == paddle::framework::proto::VarType::FP16) {
      std::normal_distribution<float> dist(static_cast<float>(mean_value),
                                           static_cast<float>(std_value));
      auto value =
          std::shared_ptr<phi::dtype::float16>(new phi::dtype::float16[size]);
      for (auto i = 0; i < size; ++i) {
        *(value.get() + i) = static_cast<phi::dtype::float16>(dist(*engine));
      }
      OpCommand::FillConstant(constant_tensor, out_dims, value.get());
    } else if (out.DType() == paddle::framework::proto::VarType::FP32) {
      std::normal_distribution<float> dist(static_cast<float>(mean_value),
                                           static_cast<float>(std_value));
      auto value = std::shared_ptr<float>(new float[size]);
      for (auto i = 0; i < size; ++i) {
        *(value.get() + i) = static_cast<float>(dist(*engine));
      }
      OpCommand::FillConstant(constant_tensor, out_dims, value.get());
    } else if (out.DType() == paddle::framework::proto::VarType::FP64) {
      std::normal_distribution<double> dist(static_cast<double>(mean_value),
                                            static_cast<double>(std_value));
      auto value = std::shared_ptr<double>(new double[size]);
      for (auto i = 0; i < size; ++i) {
        *(value.get() + i) = static_cast<double>(dist(*engine));
      }
      OpCommand::FillConstant(constant_tensor, out_dims, value.get());
    } else {
      graph::utils::log() << "[ERROR] fill_constant unsupported datatype "
                          << out.DType() << std::endl;
      exit(-1);
    }

    if (out.op() == nullptr) {
      out = constant_tensor;
    } else {
      OpCommand("Assign").Input(out).Input(constant_tensor);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(uniform_random, custom_graph::UniformRandomAdapter);
REG_OP_ADAPTER(gaussian_random, custom_graph::GaussianRandomAdapter);
