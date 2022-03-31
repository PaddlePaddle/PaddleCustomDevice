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

#pragma once

#include "paddle/phi/extension.h"

#include "boost/config.hpp"
#include "boost/variant.hpp"

#include "acl/acl.h"

aclDataType ConvertToNpuDtype(paddle::experimental::DataType dtype);
aclFormat ConvertToNpuFormat(paddle::experimental::DataLayout layout);

using NPUAttribute = boost::variant<boost::blank,
                                    int,
                                    float,
                                    std::string,
                                    std::vector<int>,
                                    std::vector<float>,
                                    std::vector<std::string>,
                                    bool,
                                    std::vector<bool>,
                                    int64_t,
                                    std::vector<int64_t>,
                                    std::vector<double>,
                                    std::vector<std::vector<int64_t>>>;

using NPUAttributeMap = std::unordered_map<std::string, NPUAttribute>;

class NpuOpRunner {
 public:
  NpuOpRunner();
  explicit NpuOpRunner(const std::string &op_type);
  NpuOpRunner(const std::string &op_type,
              const std::vector<phi::DenseTensor> &inputs = {},
              const std::vector<phi::DenseTensor> &outputs = {},
              const NPUAttributeMap &attrs = {});

  NpuOpRunner(const NpuOpRunner &runner) = delete;
  NpuOpRunner &operator=(const NpuOpRunner &runner) = delete;

  ~NpuOpRunner();

  const std::string &Type();

  NpuOpRunner &SetType(const std::string &name);

  NpuOpRunner &AddAttr(const std::string &name, const NPUAttribute &attr);

  NpuOpRunner &AddAttrDataType(const std::string &name,
                               const NPUAttribute &attr);

  NpuOpRunner &AddAttrs(const NPUAttributeMap &attrs);

  NpuOpRunner &AddInput(const phi::DenseTensor &tensor);

  NpuOpRunner &AddInput(const phi::DenseTensor &tensor, aclMemType mem_type);

  NpuOpRunner &AddInput(std::vector<int32_t> &&dims);

  NpuOpRunner &AddInput(std::vector<int64_t> &&dims);

  NpuOpRunner &AddInput(std::vector<float> &&values);

  NpuOpRunner &AddInput(std::vector<double> &&values);

  NpuOpRunner &AddOutput(const phi::DenseTensor &tensor);

  NpuOpRunner &AddInputs(const std::vector<phi::DenseTensor> &tensors);

  NpuOpRunner &AddInputNames(const std::vector<std::string> &names);

  NpuOpRunner &AddOutputs(const std::vector<phi::DenseTensor> &tensors);

  aclTensorDesc *GetInputDesc(size_t index);

  aclTensorDesc *GetOutputDesc(size_t index);

  std::vector<aclTensorDesc *> &GetInputDescs();

  std::vector<aclTensorDesc *> &GetOutputDescs();

  std::vector<aclDataBuffer *> &GetInputBuffers();

  std::vector<aclDataBuffer *> &GetOutputBuffers();

  void Run(aclrtStream stream = nullptr, bool sync = false) const;

  static void TypeAdapter(
      const std::vector<phi::DenseTensor> &inputs,
      const std::vector<phi::DenseTensor> &outputs,
      const NPUAttributeMap &attrs,
      const phi::CustomContext &dev_ctx,
      std::function<void(const std::vector<phi::DenseTensor> &,
                         const std::vector<phi::DenseTensor> &,
                         const NPUAttributeMap &,
                         const phi::CustomContext &)> op_runner,
      const std::vector<paddle::experimental::DataType> &input_type,
      const std::vector<paddle::experimental::DataType> &output_type) {
    std::vector<phi::DenseTensor> tmp_inputs(inputs.size());
    std::vector<phi::DenseTensor> tmp_outputs(outputs.size());

    for (size_t i = 0; i < input_type.size(); ++i) {
      bool cast_input =
          (input_type[i] == paddle::experimental::DataType::UNDEFINED ||
           input_type[i] != inputs[i].dtype());
      if (!cast_input) {
        tmp_inputs[i].ShareDataWith(inputs[i]);
      } else {
        tmp_inputs[i].Resize(inputs[i].dims());
        tmp_inputs[i].mutable_data(dev_ctx.GetPlace(), input_type[i]);

        const auto &cast_runner = NpuOpRunner(
            "Cast",
            {inputs[i]},
            {tmp_inputs[i]},
            {{"dst_type", static_cast<int>(ConvertToNpuDtype(input_type[i]))}});
        cast_runner.Run(dev_ctx.stream());
      }
    }
    for (size_t i = 0; i < output_type.size(); ++i) {
      bool cast_output =
          (output_type[i] == paddle::experimental::DataType::UNDEFINED ||
           output_type[i] != outputs[i].dtype());
      if (!cast_output) {
        tmp_outputs[i].ShareDataWith(outputs[i]);
      } else {
        tmp_outputs[i].Resize(outputs[i].dims());
        tmp_outputs[i].mutable_data(dev_ctx.GetPlace(), output_type[i]);
      }
    }

    op_runner(tmp_inputs, tmp_outputs, attrs, dev_ctx);

    for (size_t i = 0; i < output_type.size(); ++i) {
      bool cast_output =
          (output_type[i] == paddle::experimental::DataType::UNDEFINED ||
           output_type[i] != outputs[i].dtype());
      if (cast_output) {
        const auto &cast_runner = NpuOpRunner(
            "Cast",
            {tmp_outputs[i]},
            {outputs[i]},
            {{"dst_type",
              static_cast<int>(ConvertToNpuDtype(outputs[i].dtype()))}});
        cast_runner.Run(dev_ctx.stream());
      }
    }
  }

 private:
  aclTensorDesc *CreateTensorDesc(phi::DenseTensor tensor,
                                  aclMemType mem_type = ACL_MEMTYPE_DEVICE);
  aclDataBuffer *CreateDataBuffer(phi::DenseTensor tensor);

 private:
  std::string op_type_;
  std::vector<aclDataBuffer *> input_buffers_;
  std::vector<aclDataBuffer *> output_buffers_;
  std::vector<aclTensorDesc *> input_descs_;
  std::vector<aclTensorDesc *> output_descs_;
  std::vector<phi::DenseTensor> host_tensors_;
  aclopAttr *attr_{nullptr};
};
