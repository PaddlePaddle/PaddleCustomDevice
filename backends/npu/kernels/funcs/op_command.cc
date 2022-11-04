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

#include "kernels/funcs/op_command.h"

#include "acl/acl_op_compiler.h"
#include "pybind11/pybind11.h"

#define RELEASE_GIL_THEN_RUN(expr)        \
  if (PyGILState_Check()) {               \
    pybind11::gil_scoped_release release; \
    expr;                                 \
  } else {                                \
    expr;                                 \
  }

namespace custom_kernel {
namespace experimental {

aclDataType ConvertToNpuDtype(paddle::experimental::DataType dtype) {
  static std::map<paddle::experimental::DataType, aclDataType>  //
      DTYPE_2_ACL_DTYPE = {
          {paddle::experimental::DataType::BOOL, ACL_BOOL},
          {paddle::experimental::DataType::UINT8, ACL_UINT8},
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
          {paddle::experimental::DataType::BOOL, ge::DataType::DT_BOOL},
          {paddle::experimental::DataType::UINT8, ge::DataType::DT_UINT8},
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

class AclCommand : public NpuCommand {
 public:
  AclCommand(const std::string &op_type) : NpuCommand(op_type) {}

  ~AclCommand() override {}

  void AddInput(Tensor &tensor) override { ins_.push_back(tensor); }

  void AddOutput(Tensor &tensor) override { outs_.push_back(tensor); }

  void AddAttribute(const std::string &name,
                    const NpuAttribute &attr) override {
    if (!attr_) {
      attr_ = aclopCreateAttr();
    }
    if (attr.type() == typeid(bool)) {
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrBool(attr_, name.c_str(), paddle::get<bool>(attr)));
    } else if (attr.type() == typeid(int)) {
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrInt(attr_, name.c_str(), paddle::get<int32_t>(attr)));
    } else if (attr.type() == typeid(int64_t)) {
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrInt(attr_, name.c_str(), paddle::get<int64_t>(attr)));
    } else if (attr.type() == typeid(float)) {
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrFloat(attr_, name.c_str(), paddle::get<float>(attr)));
    } else if (attr.type() == typeid(std::vector<bool>)) {
      auto a = paddle::get<std::vector<bool>>(attr);
      std::vector<uint8_t> cast_a;
      for (auto it : a) {
        cast_a.push_back(static_cast<uint8_t>(it));
      }
      PADDLE_ENFORCE_NPU_SUCCESS(aclopSetAttrListBool(
          attr_, name.c_str(), cast_a.size(), cast_a.data()));
    } else if (attr.type() == typeid(std::vector<int32_t>)) {
      auto a = paddle::get<std::vector<int32_t>>(attr);
      std::vector<int64_t> cast_a;
      for (auto it : a) {
        cast_a.push_back(static_cast<int64_t>(it));
      }
      PADDLE_ENFORCE_NPU_SUCCESS(aclopSetAttrListInt(
          attr_, name.c_str(), cast_a.size(), cast_a.data()));
    } else if (attr.type() == typeid(std::vector<int64_t>)) {
      auto a = paddle::get<std::vector<int64_t>>(attr);
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrListInt(attr_, name.c_str(), a.size(), a.data()));
    } else if (attr.type() == typeid(std::vector<float>)) {
      auto a = paddle::get<std::vector<float>>(attr);
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrListFloat(attr_, name.c_str(), a.size(), a.data()));
    } else if (attr.type() == typeid(std::string)) {
      auto a = paddle::get<std::string>(attr);
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrString(attr_, name.c_str(), a.c_str()));
    } else if (attr.type() == typeid(std::vector<std::string>)) {
      auto a = paddle::get<std::vector<std::string>>(attr);
      std::vector<const char *> s;
      for (auto &it : a) {
        s.push_back(it.data());
      }
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrListString(attr_, name.c_str(), s.size(), s.data()));
    } else if (attr.type() == typeid(std::vector<std::vector<int64_t>>)) {
      auto a = paddle::get<std::vector<std::vector<int64_t>>>(attr);
      std::vector<int64_t *> data;
      std::vector<int> num;
      for (auto &&v : a) {
        data.push_back(v.data());
        num.push_back(v.size());
      }
      PADDLE_ENFORCE_NPU_SUCCESS(aclopSetAttrListListInt(
          attr_, name.c_str(), data.size(), num.data(), data.data()));
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Can not convert attribubte '%s' to convert to aclopAttr", name));
    }
  }

  void Run(const phi::CustomContext &ctx) override {
    auto stream = ctx.stream();

    PADDLE_ENFORCE_NOT_NULL(
        stream,
        phi::errors::External("Stream should not be null, please check."));

    VLOG(5) << "NpuOpRunner(" << this << ") Run:";
    VLOG(4) << "op_type: " << op_type_;
    VLOG(4) << "input_desc.size: " << ins_.size();
    VLOG(4) << "output_desc.size: " << outs_.size();
    VLOG(4) << "attr: " << attr_;
    VLOG(4) << "stream: " << stream;

    std::vector<aclDataBuffer *> input_buffers_;
    std::vector<aclDataBuffer *> output_buffers_;
    std::vector<aclTensorDesc *> input_descs_;
    std::vector<aclTensorDesc *> output_descs_;
    for (auto &item : ins_) {
      input_buffers_.push_back(item.buffer_);
      input_descs_.push_back(item.desc_);
    }
    for (auto &item : outs_) {
      output_buffers_.push_back(item.buffer_);
      output_descs_.push_back(item.desc_);
    }

    aclError ret;

    // Ensure that the Gil has been released before running
    // aclopCompileAndExecute.
    RELEASE_GIL_THEN_RUN({
      ret = aclopCompileAndExecute(op_type_.c_str(),
                                   input_descs_.size(),
                                   input_descs_.data(),
                                   input_buffers_.data(),
                                   output_descs_.size(),
                                   output_descs_.data(),
                                   output_buffers_.data(),
                                   attr_,
                                   ACL_ENGINE_SYS,
                                   ACL_COMPILE_SYS,
                                   NULL,
                                   stream);
    });
    VLOG(4) << "after aclopCompileAndExecute: " << ret;
    //   ret = aclrtSynchronizeStream(stream);
    //   VLOG(4) << "after aclrtSynchronizeStream: " << ret;
    PADDLE_ENFORCE_NPU_SUCCESS(ret);
  }

 private:
  std::vector<Tensor> ins_;
  std::vector<Tensor> outs_;
  aclopAttr *attr_{nullptr};
};

// graph command

std::unordered_map<phi::DenseTensor *, Tensor> global_graph_tensor_map;

Tensor &GetGraphTensor(const phi::DenseTensor &tensor) {
  return global_graph_tensor_map[const_cast<phi::DenseTensor *>(&tensor)];
}

class GraphCommand : public NpuCommand {
 public:
  GraphCommand(const std::string &op_type) : NpuCommand(op_type) {
    node_ = std::make_shared<IrNode>(op_type);
  }

  ~GraphCommand() override {}

  void AddInput(Tensor &tensor) override {
    in_index_++;
    if (tensor.in_) {
      OperatorSetInput(node_->ge_op_,
                       node_->ins_.size(),
                       tensor.in_->ge_op_,
                       tensor.in_index_);
    } else {
      tensor.is_input_ = true;
    }
    tensor.outs_.push_back(node_.get());
  }

  void AddOutput(Tensor &tensor) override {
    // t->in_index_ = node_->outs_.size();
    tensor.in_index_ = out_index_++;
    tensor.in_ = node_;

    if (node_->ins_.size() == 0) {
      tensor.is_input_ = true;
    }
  }

  void AddAttribute(const std::string &name,
                    const NpuAttribute &attr) override {
    if (attr.type() == typeid(bool)) {
      OperatorSetAttrBool(node_->ge_op_, name.c_str(), paddle::get<bool>(attr));
    } else if (attr.type() == typeid(int)) {
      OperatorSetAttrInt32(
          node_->ge_op_, name.c_str(), paddle::get<int32_t>(attr));
    } else if (attr.type() == typeid(int64_t)) {
      OperatorSetAttrInt64(
          node_->ge_op_, name.c_str(), paddle::get<int64_t>(attr));
    } else if (attr.type() == typeid(float)) {
      OperatorSetAttrFloat(
          node_->ge_op_, name.c_str(), paddle::get<float>(attr));
    } else if (attr.type() == typeid(std::vector<bool>)) {
      auto a = paddle::get<std::vector<bool>>(attr);
      std::vector<uint8_t> cast_a;
      for (auto it : a) {
        cast_a.push_back(static_cast<uint8_t>(it));
      }
      OperatorSetAttrBoolList(
          node_->ge_op_, name.c_str(), cast_a.data(), cast_a.size());
    } else if (attr.type() == typeid(std::vector<int32_t>)) {
      auto a = paddle::get<std::vector<int32_t>>(attr);
      OperatorSetAttrInt32List(node_->ge_op_, name.c_str(), a.data(), a.size());
    } else if (attr.type() == typeid(std::vector<int64_t>)) {
      auto a = paddle::get<std::vector<int64_t>>(attr);
      OperatorSetAttrInt64List(node_->ge_op_, name.c_str(), a.data(), a.size());
    } else if (attr.type() == typeid(std::vector<float>)) {
      auto a = paddle::get<std::vector<float>>(attr);
      OperatorSetAttrFloatList(node_->ge_op_, name.c_str(), a.data(), a.size());
    } else if (attr.type() == typeid(std::string)) {
      auto a = paddle::get<std::string>(attr);
      OperatorSetAttrString(node_->ge_op_, name.c_str(), a.data());
    } else if (attr.type() == typeid(std::vector<std::string>)) {
      auto a = paddle::get<std::vector<std::string>>(attr);
      std::vector<const char *> s;
      for (auto &it : a) {
        s.push_back(it.data());
      }
      OperatorSetAttrStringList(
          node_->ge_op_, name.c_str(), s.data(), s.size());
    } else if (attr.type() == typeid(C_GE_Tensor *)) {
      OperatorSetAttrTensor(
          node_->ge_op_, name.c_str(), paddle::get<C_GE_Tensor *>(attr));
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Can not convert attribubte '%s' to convert to OperatorAttr", name));
    }
  }

  void Run(const phi::CustomContext &) override {}

  static Tensor &GetTensor(const phi::DenseTensor &tensor);

 private:
  std::shared_ptr<IrNode> node_{nullptr};
  size_t in_index_;
  size_t out_index_;
  std::vector<Tensor> storage_;
};

OpCommand::OpCommand(const std::string &op_type) {
  ACL_RUN(cmd_ = std::make_shared<AclCommand>(op_type));
  GRAPH_RUN(cmd_ = std::make_shared<GraphCommand>(op_type));
}

OpCommand &OpCommand::Input(const phi::DenseTensor &tensor) {
  ACL_RUN({
    Tensor t(tensor);
    cmd_->AddInput(t);
    return *this;
  });
  cmd_->AddInput(GetGraphTensor(tensor));
  return *this;
}

OpCommand &OpCommand::Output(const phi::DenseTensor &tensor) {
  ACL_RUN({
    Tensor t(tensor);
    cmd_->AddOutput(t);
    return *this;
  });
  cmd_->AddOutput(GetGraphTensor(tensor));
  return *this;
}

void TensorHelper::Assign(const phi::DenseTensor &dst,
                          const phi::DenseTensor &src) {
  auto &dst_tensor = GetGraphTensor(dst);
  auto &src_tensor = GetGraphTensor(src);
  dst_tensor = src_tensor;
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
    std::vector<C_GE_Operator *> target_ops;

    for (auto i = 0; i < feed_tensor_num; ++i) {
      auto dense_tensor =
          reinterpret_cast<phi::DenseTensor *>(feed_tensor_name[i]);
      auto &graph_tensor =
          custom_kernel::experimental::global_graph_tensor_map[dense_tensor];
      auto node = std::make_shared<custom_kernel::experimental::IrNode>("Data");
      graph_tensor.in_ = node;
      graph_tensor.in_index_ = 0;

      for (auto &t : graph_tensor.outs_) {
        OperatorSetInput(
            t->ge_op_,
            std::find(t->ins_.begin(), t->ins_.end(), &graph_tensor) -
                t->ins_.begin(),
            graph_tensor.in_->ge_op_,
            graph_tensor.in_index_);
      }

      input_ops.push_back(node->ge_op_);
    }

    for (auto &item : custom_kernel::experimental::global_graph_tensor_map) {
      if (item.second.is_input_ &&
          std::find(feed_tensor_name,
                    feed_tensor_name + feed_tensor_num,
                    reinterpret_cast<char *>(item.first)) ==
              feed_tensor_name + feed_tensor_num) {
        input_ops.push_back(item.second.in_->ge_op_);
        // auto dims = phi::vectorize(item.first->dims());
        // OperatorUpdateInputDesc(
        //     item.second.in_->ge_op_,
        //     "x",
        //     dims.data(),
        //     dims.size(),
        //     custom_kernel::experimental::ConvertToGEDtype(item.first->dtype()),
        //     custom_kernel::experimental::ConvertToGEFormat(
        //         item.first->layout()));

        // OperatorUpdateOutputDesc(
        //     item.second.in_->ge_op_,
        //     "y",
        //     dims.data(),
        //     dims.size(),
        //     custom_kernel::experimental::ConvertToGEDtype(item.first->dtype()),
        //     custom_kernel::experimental::ConvertToGEFormat(
        //         item.first->layout()));
      }
    }

    for (auto i = 0; i < fetch_tensor_num; ++i) {
      auto dense_tensor =
          reinterpret_cast<phi::DenseTensor *>(fetch_tensor_name[i]);
      auto &graph_tensor =
          custom_kernel::experimental::global_graph_tensor_map[dense_tensor];
      output_ops.push_back(graph_tensor.in_->ge_op_);
    }

    std::cerr << "input_ops size=" << input_ops.size() << std::endl;
    std::cerr << "output_ops size=" << output_ops.size() << std::endl;
    GraphSetInput(
        graph_cache[c_graph].graph, input_ops.data(), input_ops.size());
    GraphSetOutput(
        graph_cache[c_graph].graph, output_ops.data(), output_ops.size());
    GraphSetTarget(
        graph_cache[c_graph].graph, target_ops.data(), target_ops.size());
    SessionAddGraph(
        session, graph_cache[c_graph].graph_id, graph_cache[c_graph].graph);
  }

  for (auto i = 0; i < feed_tensor_num; ++i) {
    auto dense_tensor =
        reinterpret_cast<phi::DenseTensor *>(feed_tensor_name[i]);
    auto dims = phi::vectorize(dense_tensor->dims());
    auto tensor = CreateTensor();
    SetTensor(tensor,
              feed_tensor_data[i],
              dims.data(),
              dims.size(),
              static_cast<int>(custom_kernel::experimental::ConvertToGEDtype(
                  dense_tensor->dtype())),
              static_cast<int>(custom_kernel::experimental::ConvertToGEFormat(
                  dense_tensor->layout())));
    input_tensors.push_back(tensor);
  }

  for (auto i = 0; i < fetch_tensor_num; ++i) {
    auto dense_tensor =
        reinterpret_cast<phi::DenseTensor *>(fetch_tensor_name[i]);
    if (custom_kernel::experimental::global_graph_tensor_map.find(
            dense_tensor) ==
        custom_kernel::experimental::global_graph_tensor_map.end()) {
      std::cerr << "[ERROR] can not found fetch tensor " << dense_tensor
                << std::endl;
      return C_FAILED;
    }
    auto &graph_tensor =
        custom_kernel::experimental::global_graph_tensor_map[dense_tensor];

    auto tensor = CreateTensor();
    output_tensors.push_back(tensor);
  }

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

  for (auto &t : input_tensors) {
    DestroyTensor(t);
  }
  for (auto &t : output_tensors) {
    DestroyTensor(t);
  }
  return C_SUCCESS;
}
