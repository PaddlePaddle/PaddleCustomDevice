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

#include "kernels/funcs/acl_command.h"
#include "kernels/funcs/graph_command.h"

namespace custom_kernel {
namespace experimental {

aclDataType ConvertToNpuDtype(paddle::experimental::DataType dtype) {
  static std::map<paddle::experimental::DataType, aclDataType>  //
      DTYPE_2_ACL_DTYPE = {
          {paddle::experimental::DataType::BOOL, ACL_BOOL},
          {paddle::experimental::DataType::UINT8, ACL_UINT8},
          {paddle::experimental::DataType::UINT16, ACL_UINT16},
          {paddle::experimental::DataType::UINT32, ACL_UINT32},
          {paddle::experimental::DataType::UINT64, ACL_UINT64},
          {paddle::experimental::DataType::INT8, ACL_INT8},
          {paddle::experimental::DataType::INT16, ACL_INT16},
          {paddle::experimental::DataType::INT32, ACL_INT32},
          {paddle::experimental::DataType::INT64, ACL_INT64},
          {paddle::experimental::DataType::FLOAT16, ACL_FLOAT16},
          {paddle::experimental::DataType::FLOAT32, ACL_FLOAT},
          {paddle::experimental::DataType::FLOAT64, ACL_DOUBLE},
      };
  auto iter = DTYPE_2_ACL_DTYPE.find(dtype);
  PADDLE_ENFORCE_NE(
      iter,
      DTYPE_2_ACL_DTYPE.end(),
      phi::errors::NotFound(
          "The data type %s can not convert to ACL data type.", dtype));
  return iter->second;
}

ge::DataType ConvertToGEDtype(paddle::experimental::DataType dtype) {
  static std::map<paddle::experimental::DataType, ge::DataType>  //
      DTYPE_2_GE_DTYPE = {
          {paddle::experimental::DataType::BOOL, ge::DataType::DT_UINT8},
          {paddle::experimental::DataType::UINT8, ge::DataType::DT_UINT8},
          {paddle::experimental::DataType::UINT16, ge::DataType::DT_UINT16},
          {paddle::experimental::DataType::UINT32, ge::DataType::DT_UINT32},
          {paddle::experimental::DataType::UINT64, ge::DataType::DT_UINT64},
          {paddle::experimental::DataType::INT8, ge::DataType::DT_INT8},
          {paddle::experimental::DataType::INT16, ge::DataType::DT_INT16},
          {paddle::experimental::DataType::INT32, ge::DataType::DT_INT32},
          {paddle::experimental::DataType::INT64, ge::DataType::DT_INT64},
          {paddle::experimental::DataType::FLOAT16, ge::DataType::DT_FLOAT16},
          {paddle::experimental::DataType::FLOAT32, ge::DataType::DT_FLOAT},
          {paddle::experimental::DataType::FLOAT64, ge::DataType::DT_DOUBLE},
      };
  auto iter = DTYPE_2_GE_DTYPE.find(dtype);
  PADDLE_ENFORCE_NE(
      iter,
      DTYPE_2_GE_DTYPE.end(),
      phi::errors::NotFound(
          "The data type %s can not convert to ACL data type.", dtype));
  return iter->second;
}

aclFormat ConvertToNpuFormat(phi::DataLayout layout) {
  static std::map<phi::DataLayout, aclFormat> DATA_LAYOUT_2_ACL_FORMAT = {
      {phi::DataLayout::NCHW, ACL_FORMAT_NCHW},
      {phi::DataLayout::NHWC, ACL_FORMAT_NHWC},
      {phi::DataLayout::kNCDHW, ACL_FORMAT_NCDHW},
      {phi::DataLayout::kNDHWC, ACL_FORMAT_NDHWC},
      {phi::DataLayout::ANY, ACL_FORMAT_ND},
  };
  auto iter = DATA_LAYOUT_2_ACL_FORMAT.find(layout);
  PADDLE_ENFORCE_NE(
      iter,
      DATA_LAYOUT_2_ACL_FORMAT.end(),
      phi::errors::NotFound(
          "The data type (%s) can not convert to ACL data type.", layout));
  return iter->second;
}

ge::Format ConvertToGEFormat(phi::DataLayout layout) {
  static std::map<phi::DataLayout, ge::Format> DATA_LAYOUT_2_GE_FORMAT = {
      {phi::DataLayout::NCHW, ge::Format::FORMAT_NCHW},
      {phi::DataLayout::NHWC, ge::Format::FORMAT_NHWC},
      {phi::DataLayout::kNCDHW, ge::Format::FORMAT_NCDHW},
      {phi::DataLayout::kNDHWC, ge::Format::FORMAT_NDHWC},
      {phi::DataLayout::ANY, ge::Format::FORMAT_ND},
  };
  auto iter = DATA_LAYOUT_2_GE_FORMAT.find(layout);
  PADDLE_ENFORCE_NE(
      iter,
      DATA_LAYOUT_2_GE_FORMAT.end(),
      phi::errors::NotFound(
          "The data type (%s) can not convert to ACL data type.", layout));
  return iter->second;
}

OpCommand::OpCommand(const std::string &op_type) {
  ACL_RUN(cmd_ = std::make_shared<AclCommand>(op_type));
  GRAPH_RUN(cmd_ = std::make_shared<GraphCommand>(op_type));
}

OpCommand &OpCommand::Input(const phi::Scalar &scalar) {
  cmd_->AddInput(scalar);
  return *this;
}

OpCommand &OpCommand::ScalarInput(const phi::DenseTensor &tensor) {
  if (tensor.place() == phi::CPUPlace()) {
    cmd_->AddHostScalarInput(tensor);
  } else {
    cmd_->AddScalarInput(tensor);
  }
  return *this;
}

OpCommand &OpCommand::Input(const phi::DenseTensor &tensor) {
  if (tensor.place() == phi::CPUPlace()) {
    cmd_->AddHostInput(tensor);
  } else {
    cmd_->AddInput(tensor);
  }
  return *this;
}

OpCommand &OpCommand::Output(phi::DenseTensor &tensor) {
  cmd_->AddOutput(tensor);
  return *this;
}

OpCommand &OpCommand::Input(const phi::DenseTensor &tensor,
                            TensorDescMaker maker) {
  if (tensor.place() == phi::CPUPlace()) {
    cmd_->AddHostInput(tensor, maker);
  } else {
    cmd_->AddInput(tensor, maker);
  }
  return *this;
}

OpCommand &OpCommand::ScalarInput(const phi::DenseTensor &tensor,
                                  TensorDescMaker maker) {
  if (tensor.place() == phi::CPUPlace()) {
    cmd_->AddHostScalarInput(tensor, maker);
  } else {
    cmd_->AddScalarInput(tensor, maker);
  }
  return *this;
}

OpCommand &OpCommand::Output(phi::DenseTensor &tensor, TensorDescMaker maker) {
  cmd_->AddOutput(tensor, maker);
  return *this;
}

OpCommand &OpCommand::Input() {
  cmd_->AddInput();
  return *this;
}
OpCommand &OpCommand::Output() {
  cmd_->AddOutput();
  return *this;
}

void OpCommandHelper::Assign(const phi::CustomContext &ctx,
                             const phi::DenseTensor &src,
                             phi::DenseTensor *dst) {
  ACL_RUN({ custom_kernel::TensorCopy(ctx, src, false, dst); });
  GRAPH_RUN({ *dst = src; });
}

void OpCommandHelper::MarkAsParameter(phi::DenseTensor *dst) {
  GRAPH_RUN({
    reinterpret_cast<TensorNode *>(dst->data())
        ->SetTag(custom_kernel::experimental::TensorNodeTag::PARAMETER);
  });
}

}  // namespace experimental
}  // namespace custom_kernel

struct GEGraph {
  size_t graph_id;
  C_GE_Graph *graph;
};

std::unordered_map<C_Graph, GEGraph> graph_cache;
bool graph_cache_hit = true;

C_Status graph_engine_prepare_graph(const C_Device device,
                                    const C_Stream stream,
                                    const C_Scope c_scope,
                                    const C_Graph c_graph) {
  VLOG(10) << "graph_engine_prepare_graph: " << stream << ", " << c_scope
           << ", " << c_graph;
  auto session = GetSession(c_scope);
  if (graph_cache.find(c_graph) == graph_cache.end()) {
    std::string graph_name = "graph_" + std::to_string(graph_cache.size());
    auto graph_id = graph_cache.size();
    graph_cache[c_graph] = {graph_id, CreateGraph(c_scope, graph_name.c_str())};
    graph_cache_hit = false;
  } else {
    graph_cache_hit = true;
  }
}

void global_variable_initializer(const C_Scope c_scope, const C_Graph c_graph) {
  std::vector<C_GE_Operator *> input_ops, output_ops;
  std::vector<size_t> output_ops_index;
  auto session = GetSession(c_scope);

  std::vector<custom_kernel::experimental::TensorNode *> parameters;
  std::vector<std::shared_ptr<custom_kernel::experimental::OpNode>> variables;
  std::vector<custom_kernel::experimental::TensorDescMaker> descs;

  for (auto tensor : custom_kernel::experimental::TensorNode::storage()) {
    if (tensor->IsParameter()) {
      auto variable_node =
          std::make_shared<custom_kernel::experimental::OpNode>("Variable");
      auto assign_node =
          std::make_shared<custom_kernel::experimental::OpNode>("Assign");

      auto desc = tensor->Node()->OutputDescs()[tensor->NodeIndex()];
      auto dims = phi::vectorize(desc.dims_);

      PADDLE_ENFORCE(
          desc.Valid(),
          phi::errors::Unavailable("Parameter %s's desc must be valid.",
                                   variable_node->Name()));

      OperatorUpdateOutputDesc(
          variable_node->GeOp(),
          "y",
          dims.data(),
          dims.size(),
          custom_kernel::experimental::ConvertToGEDtype(desc.dtype_),
          custom_kernel::experimental::ConvertToGEFormat(desc.layout_));

      OperatorSetInput(assign_node->GeOp(), 0, variable_node->GeOp(), 0);
      OperatorUpdateInputDesc(
          assign_node->GeOp(),
          "ref",
          dims.data(),
          dims.size(),
          custom_kernel::experimental::ConvertToGEDtype(desc.dtype_),
          custom_kernel::experimental::ConvertToGEFormat(desc.layout_));

      OperatorSetInput(assign_node->GeOp(), 1, tensor->NodeGeOp(), 0);
      OperatorUpdateInputDesc(
          assign_node->GeOp(),
          "value",
          dims.data(),
          dims.size(),
          custom_kernel::experimental::ConvertToGEDtype(desc.dtype_),
          custom_kernel::experimental::ConvertToGEFormat(desc.layout_));

      input_ops.push_back(tensor->NodeGeOp());
      variables.push_back(variable_node);
      parameters.push_back(tensor.get());
      descs.push_back(desc);
    }
  }

  if (input_ops.size()) {
    GraphSetInput(
        graph_cache[c_graph].graph, input_ops.data(), input_ops.size());
    GraphSetOutput(graph_cache[c_graph].graph, nullptr, nullptr, 0);
    SessionAddGraph(
        session, graph_cache[c_graph].graph_id, graph_cache[c_graph].graph);
    SessionRunGraph(
        session, graph_cache[c_graph].graph_id, nullptr, 0, nullptr, 0);

    std::string graph_name = "graph_" + std::to_string(graph_cache.size());
    auto graph_id = graph_cache.size();
    graph_cache[c_graph] = {graph_id, CreateGraph(c_scope, graph_name.c_str())};
  }

  for (auto i = 0; i < parameters.size(); ++i) {
    variables[i]->AddOutput(parameters[i], descs[i]);
    parameters[i]->ResetFromOther(variables[i], 0);
  }
}

C_Status graph_engine_execute_graph(const C_Device device,
                                    const C_Stream stream,
                                    const C_Scope c_scope,
                                    const C_Graph c_graph,
                                    char **feed_tensor_name,
                                    void **feed_tensor_data,
                                    size_t feed_tensor_num,
                                    char **fetch_tensor_name,
                                    void **fetch_tensor_data,
                                    size_t fetch_tensor_num) {
  auto session = GetSession(c_scope);

  std::vector<C_GE_Tensor *> input_tensors;
  std::vector<C_GE_Tensor *> output_tensors;

  if (!graph_cache_hit) {
    std::vector<C_GE_Operator *> input_ops;
    std::vector<C_GE_Operator *> output_ops;
    std::vector<size_t> output_ops_index;
    std::vector<C_GE_Operator *> target_ops;

    for (auto i = 0; i < feed_tensor_num; ++i) {
      auto dense_tensor =
          reinterpret_cast<phi::DenseTensor *>(feed_tensor_name[i]);
      auto graph_tensor =
          reinterpret_cast<custom_kernel::experimental::TensorNode *>(
              dense_tensor->data());

      // NOTE(wangran16): run graph failed when the Data node not be linked to
      // other node.
      if (graph_tensor->Links().size()) {
        graph_tensor->SetTag(custom_kernel::experimental::TensorNodeTag::IN);
        auto data_node =
            std::make_shared<custom_kernel::experimental::OpNode>("Data");
        data_node->AddOutput(
            graph_tensor,
            custom_kernel::experimental::TensorDescMaker("y", *dense_tensor)
                .SetDataLayout(phi::DataLayout::ANY));
        graph_tensor->FromOther(data_node, 0);
        LOG(INFO) << "tensor: " << dense_tensor
                  << " insert node: " << data_node->Name();
      }
    }

    global_variable_initializer(c_scope, c_graph);

    for (auto tensor : custom_kernel::experimental::TensorNode::storage()) {
      LOG(INFO) << "tensor: " << tensor
                << ", has node: " << !tensor->WithoutNode()
                << ", tag: " << tensor->Tag() << ", node: " << tensor->Node()
                << ", type: "
                << (tensor->WithoutNode() ? "None" : tensor->Node()->Type())
                << ", name: "
                << (tensor->WithoutNode() ? "None" : tensor->Node()->Name())
                << "\n\n";

      if (!tensor->WithoutNode()) {
        tensor->Node()->UpdateOutputDesc(tensor->NodeIndex());
        auto desc = tensor->Node()->OutputDescs()[tensor->NodeIndex()];
        if (desc.Valid()) {
          LOG(INFO) << "\tupdate " << tensor->Node()->Name() << " output "
                    << desc.desc_name_ << " dims=" << desc.dims_
                    << ", dtype=" << desc.dtype_ << ", layout=" << desc.layout_
                    << "\n\n";
        }
      }

      for (auto link : tensor->Links()) {
        LOG(INFO) << "\tlink " << tensor->Node()->Name() << " -> "
                  << link.first->Name() << std::endl;
        auto link_input = tensor->NodeGeOp();
        auto link_input_index = tensor->NodeIndex();

        link.first->UpdateInputDesc(link.second);
        auto desc = link.first->InputDescs()[link.second];
        if (desc.Valid()) {
          LOG(INFO) << "\tupdate " << link.first->Name() << " input "
                    << desc.desc_name_ << " dims=" << desc.dims_
                    << ", dtype=" << desc.dtype_ << ", layout=" << desc.layout_
                    << std::endl;
          auto origin_dims = phi::make_ddim(tensor->NodeShape<int64_t>());
          if (origin_dims != desc.dims_ && origin_dims.size()) {
            LOG(INFO) << "\treshape " << origin_dims << " -> " << desc.dims_;
            // x -> reshape -> op
            auto shape =
                std::make_shared<custom_kernel::experimental::OpNode>("Const");
            auto shape_ge_tensor = custom_kernel::experimental::GETensorHelper::
                ConvertVectorToGETensor<int32_t>(
                    phi::vectorize<int32_t>(desc.dims_));
            OperatorSetAttrTensor(shape->GeOp(), "value", shape_ge_tensor);

            auto reshape =
                std::make_shared<custom_kernel::experimental::OpNode>(
                    "Reshape");
            OperatorSetInput(
                reshape->GeOp(), 0, tensor->NodeGeOp(), tensor->NodeIndex());
            OperatorSetInput(reshape->GeOp(), 1, shape->GeOp(), 0);

            link_input = reshape->GeOp();
            link_input_index = 0;
          }
        }
        OperatorSetInput(
            link.first->GeOp(), link.second, link_input, link_input_index);
        LOG(INFO) << std::endl;
      }
    }

    for (auto i = 0; i < feed_tensor_num; ++i) {
      auto dense_tensor =
          reinterpret_cast<phi::DenseTensor *>(feed_tensor_name[i]);
      auto graph_tensor =
          reinterpret_cast<custom_kernel::experimental::TensorNode *>(
              dense_tensor->data());
      if (!graph_tensor->WithoutNode() && graph_tensor->Links().size()) {
        LOG(INFO) << "graph add input: " << graph_tensor << ", "
                  << graph_tensor->Node()->Name() << ", "
                  << graph_tensor->NodeGeOp();
        input_ops.push_back(graph_tensor->NodeGeOp());
      }
    }

    for (auto graph_tensor :
         custom_kernel::experimental::TensorNode::storage()) {
      if (graph_tensor->IsInput() &&
          std::find(input_ops.cbegin(),
                    input_ops.cend(),
                    graph_tensor->NodeGeOp()) == input_ops.cend()) {
        LOG(INFO) << "graph add input: " << graph_tensor << ", "
                  << graph_tensor->Node()->Name() << ", "
                  << graph_tensor->NodeGeOp();
        input_ops.push_back(graph_tensor->NodeGeOp());
      }
    }

    for (auto i = 0; i < fetch_tensor_num; ++i) {
      auto dense_tensor =
          reinterpret_cast<phi::DenseTensor *>(fetch_tensor_name[i]);
      auto graph_tensor =
          reinterpret_cast<custom_kernel::experimental::TensorNode *>(
              dense_tensor->data());

      LOG(INFO) << "graph add output: " << graph_tensor << ", "
                << graph_tensor->Node()->Name() << ", "
                << graph_tensor->NodeGeOp();
      output_ops.push_back(graph_tensor->NodeGeOp());
      output_ops_index.push_back(graph_tensor->NodeIndex());
    }

    LOG(INFO) << "input_ops size=" << input_ops.size() << std::endl;
    LOG(INFO) << "output_ops size=" << output_ops.size() << std::endl;
    if (input_ops.size() == 0) {
      return C_FAILED;
    }
    VLOG(10) << "Set inputs: ";
    for (auto in : input_ops) {
      VLOG(10) << "in: " << in << ", type: " << OperatorGetOpType(in)
               << ", name: " << OperatorGetOpName(in);
    }
    GraphSetInput(
        graph_cache[c_graph].graph, input_ops.data(), input_ops.size());
    for (auto in : output_ops) {
      VLOG(10) << "out: " << in << ", type: " << OperatorGetOpType(in)
               << ", name: " << OperatorGetOpName(in);
    }
    GraphSetOutput(graph_cache[c_graph].graph,
                   output_ops.data(),
                   output_ops_index.data(),
                   output_ops.size());
    GraphSetTarget(
        graph_cache[c_graph].graph, target_ops.data(), target_ops.size());
    SessionAddGraph(
        session, graph_cache[c_graph].graph_id, graph_cache[c_graph].graph);
  }

  for (auto i = 0; i < feed_tensor_num; ++i) {
    auto dense_tensor =
        reinterpret_cast<phi::DenseTensor *>(feed_tensor_name[i]);
    auto graph_tensor =
        reinterpret_cast<custom_kernel::experimental::TensorNode *>(
            dense_tensor->data());
    if (!graph_tensor->WithoutNode() && graph_tensor->Links().size()) {
      auto dims = phi::vectorize(dense_tensor->dims());
      auto tensor = CreateTensor();
      PADDLE_ENFORCE(tensor, phi::errors::Fatal("CreateTensor failed."));
      SetTensor(
          tensor,
          feed_tensor_data[i],
          dims.data(),
          dims.size(),
          custom_kernel::experimental::ConvertToGEDtype(dense_tensor->dtype()),
          custom_kernel::experimental::ConvertToGEFormat(
              dense_tensor->layout()));
      input_tensors.push_back(tensor);
    }
  }

  for (auto i = 0; i < fetch_tensor_num; ++i) {
    auto tensor = CreateTensor();
    PADDLE_ENFORCE(tensor, phi::errors::Fatal("CreateTensor failed."));
    output_tensors.push_back(tensor);
  }

  VLOG(10) << "SessionRunGraph session=" << session
           << ", graph_id=" << graph_cache[c_graph].graph_id
           << ", input_tensors.size()=" << input_tensors.size();
  SessionRunGraph(session,
                  graph_cache[c_graph].graph_id,
                  input_tensors.data(),
                  input_tensors.size(),
                  output_tensors.data(),
                  output_tensors.size());

  for (auto i = 0; i < fetch_tensor_num; ++i) {
    memcpy(fetch_tensor_data[i],
           TensorGetData(output_tensors[i]),
           TensorGetSize(output_tensors[i]));
  }
  for (auto t : input_tensors) {
    VLOG(10) << "DestroyTensor: " << t;
    DestroyTensor(t);
  }
  for (auto t : output_tensors) {
    VLOG(10) << "DestroyTensor: " << t;
    DestroyTensor(t);
  }
  return C_SUCCESS;
}

C_Status graph_engine_initialize(const C_Device device, const C_Stream stream) {
  graph_initialize(device, stream);
  return C_SUCCESS;
}

C_Status graph_engine_finalize(const C_Device device, const C_Stream stream) {
  custom_kernel::experimental::TensorNode::storage().clear();
  graph_finalize(device, stream);
  return C_SUCCESS;
}

C_Status graph_engine_allocator_allocate(const C_Device device,
                                         void **ptr,
                                         size_t byte_size) {
  *ptr = custom_kernel::experimental::TensorNode::malloc();
  VLOG(10) << "graph_engine_allocator_allocate: ptr=" << *ptr
           << ", byte_size=" << byte_size;
  return C_SUCCESS;
}

C_Status graph_engine_allocator_deallocate(const C_Device device,
                                           void *ptr,
                                           size_t byte_size) {
  // VLOG(10) << "graph_engine_allocator_deallocate: ptr=" << ptr
  //          << ", byte_size=" << byte_size;
  // delete reinterpret_cast<custom_kernel::experimental::TensorNode *>(ptr);
  return C_SUCCESS;
}
