// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "backend/executor/cast_runner.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "backend/equivalence_trans/all_ops.h"
#include "backend/executor/gcu_node.h"
#include "backend/executor/tops_compiler.h"
#include "backend/utils/gcu_op_desc.h"
#include "backend/utils/utils.h"
#include "gcu/hlir_builder/hlir_builder.h"
#include "runtime/runtime.h"

namespace backend {
namespace {
const char* const kRunningModeSerial = "serial";
}  // namespace

using TensorNameMap = std::map<std::string, std::vector<std::string>>;
using GcuAttributeMap = backend::GcuAttributeMap;
using GcuOpDesc = backend::GcuOpDesc;
using GcuAttribute = backend::GcuAttribute;
using EquivalenceTransformer = backend::EquivalenceTransformer;
using TransformUtil = backend::TransformUtil;
using DataType = phi::DataType;

class CastExecutor {
 public:
  CastExecutor(topsExecutable_t exec,
               const GcuNode& input_nodes,
               const GcuNode& output_nodes);
  CastExecutor() = delete;
  ~CastExecutor() = default;
  CastExecutor(const CastExecutor& exec) = default;
  CastExecutor& operator=(const CastExecutor& exec) = default;
  void ReleaseResource();
  void RunGcuOp(const topsStream_t stream,
                const std::vector<int64_t>& dims,
                const DataType src_data_type,
                const DataType dst_data_type,
                const void* src_buf,
                void* dst_buf);

 private:
  topsExecutable_t tops_exec_ = nullptr;
  GcuNode input_node_;
  GcuNode output_node_;
};

class CastExecutorManager {
 public:
  ~CastExecutorManager() { ReleaseAll(); }
  void ReleaseAll() {
    for (const auto& p : cast_executors_) {
      p.second->ReleaseResource();
    }
    cast_executors_.clear();
  }
  void Add(const std::string& key, const std::shared_ptr<CastExecutor>& exec) {
    cast_executors_[key] = exec;
  }
  std::shared_ptr<CastExecutor> Find(const std::string& key) {
    if (cast_executors_.count(key) == 0) {
      return nullptr;
    }
    auto exec = cast_executors_[key];
    PADDLE_ENFORCE_NE(
        exec, nullptr, phi::errors::NotFound("buffered exec is nullptr"));
    return exec;
  }

 public:
  static CastExecutorManager* GetInstance() {
    static CastExecutorManager manager;
    return &manager;
  }

 private:
  std::map<std::string, std::shared_ptr<CastExecutor>> cast_executors_;
};

CastExecutor::CastExecutor(topsExecutable_t exec,
                           const GcuNode& input_node,
                           const GcuNode& output_node) {
  PADDLE_ENFORCE_NOT_NULL(
      exec, phi::errors::InvalidArgument("Expect executable is not null."));
  tops_exec_ = exec;
  input_node_ = input_node;
  output_node_ = output_node;
}

void CastExecutor::ReleaseResource() {
  RT_CHECK(topsDestroyExecutable(tops_exec_));
  tops_exec_ = nullptr;
}

void CastExecutor::RunGcuOp(const topsStream_t stream,
                            const std::vector<int64_t>& dims,
                            const DataType src_data_type,
                            const DataType dst_data_type,
                            const void* src_buf,
                            void* dst_buf) {
  std::vector<void*> dev_inputs;
  dev_inputs.emplace_back(const_cast<void*>(src_buf));
  std::vector<void*> dev_outputs;
  dev_outputs.emplace_back(dst_buf);

  GcuNode input_node(phi::make_ddim(dims), src_data_type);
  PADDLE_ENFORCE_EQ(
      input_node,
      input_node_,
      phi::errors::InvalidArgument(
          "input desc not equal cached, input desc: %s vs cached desc %s",
          input_node.to_str(),
          input_node_.to_str()));

  GcuNode output_node(phi::make_ddim(dims), dst_data_type);
  PADDLE_ENFORCE_EQ(
      output_node,
      output_node_,
      phi::errors::InvalidArgument("output desc not equal cached, "
                                   "input desc: %s vs cached desc %s",
                                   output_node.to_str(),
                                   output_node_.to_str()));

  RT_CHECK(topsLaunchExecutable(tops_exec_,
                                nullptr,
                                dev_inputs.data(),
                                dev_inputs.size(),
                                nullptr,
                                nullptr,
                                dev_outputs.data(),
                                dev_outputs.size(),
                                nullptr,
                                nullptr,
                                stream));

  // RT_CHECK(topsStreamSynchronize(stream));
}

void CompileExecutable(const std::string& signature,
                       const DataType src_data_type,
                       const DataType dst_data_type,
                       std::vector<int64_t> dims) {  // NOLINT
  std::string op_type = "cast";
  VLOG(10) << "OpType " << op_type << " strart compile. ";

  // build input output attrs
  TensorNameMap input_names;
  input_names["X"] = {"x"};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  GcuAttributeMap attrs;
  attrs["in_dtype"] = static_cast<int>(src_data_type);
  attrs["out_dtype"] = static_cast<int>(dst_data_type);

  GcuBuilderPtr builder = std::make_shared<GcuBuilder>();
  PADDLE_ENFORCE_NE(
      builder,
      nullptr,
      phi::errors::Fatal("builfer is nullptr, graph:%s", signature.c_str()));
  builder->SetShapeInference(true);

  auto func =
      EquivalenceTransformer::GetInstance().Get(op_type, backend::INSENSITIVE);

  PADDLE_ENFORCE_NE(
      func,
      nullptr,
      phi::errors::NotFound(
          "OpType %s is not register gcu op convert func, please check.",
          op_type.c_str()));

  // build input gcu op
  auto ptype = TransformUtil::ConvertDataType(src_data_type);
  builder::Type input_type(dims, ptype);
  auto input_gcu_op = std::make_shared<GcuOp>(builder->CreateInput(input_type));

  std::map<std::string, std::vector<GcuOpPtr>> map_inputs;
  map_inputs["X"] = {input_gcu_op};

  auto op_desc = std::make_shared<GcuOpDesc>(
      GcuOpDesc(op_type, input_names, output_names, attrs));
  VLOG(10) << "Transfered to gcu node start, type:" << op_type;
  GcuOpPtr output_gcu_op =
      func(builder, op_desc, map_inputs, kRunningModeSerial);
  VLOG(10) << "Transfered to gcu node end, type:" << op_type;

  PADDLE_ENFORCE_NE(output_gcu_op,
                    nullptr,
                    phi::errors::Fatal(
                        "op type:%s transfered gcu node should not be nullptr!",
                        op_type.c_str()));

  builder->SetOutput({*output_gcu_op});

  VLOG(6) << "Convert to gcu graph finished!";
  builder->Dump();
  auto hlir_module = builder->GetModule();

  // compile
  topsExecutable_t tops_executable = CompileTopsExecutable(hlir_module);
  VLOG(6) << "Compiler CompileHLIR end for program " << signature;

  GcuNode input_node(phi::make_ddim(dims), src_data_type);
  GcuNode output_nodes(phi::make_ddim(dims), dst_data_type);

  auto manager = CastExecutorManager::GetInstance();
  auto gcu_exec =
      std::make_shared<CastExecutor>(tops_executable, input_node, output_nodes);
  manager->Add(signature, gcu_exec);
}

void RunExecutableAsync(const std::string signature,
                        const topsStream_t stream,
                        const std::vector<int64_t> dims,
                        const DataType src_data_type,
                        const DataType dst_data_type,
                        const void* src_buf,
                        void* dst_buf) {
  VLOG(3) << "=== start RunExecutableSync ===";

  auto manager = CastExecutorManager::GetInstance();
  auto gcu_exec = manager->Find(signature);

  PADDLE_ENFORCE_NOT_NULL(
      gcu_exec,
      phi::errors::NotFound(" Not find executor for signature:%s",
                            signature.c_str()));

  gcu_exec->RunGcuOp(
      stream, dims, src_data_type, dst_data_type, src_buf, dst_buf);

  VLOG(3) << "=== end RunExecutableSync ===";
}

void CastRunner(const topsStream_t stream,
                const std::vector<int64_t> dims,
                const DataType src_data_type,
                const DataType dst_data_type,
                const void* src_buf,
                void* dst_buf) {
  VLOG(6) << "start cast runner ";
  auto build_signature = [&]() -> std::string {
    std::ostringstream os;
    os << "input type : " << phi::DataTypeToString(src_data_type) << ", ";
    os << "output type : " << phi::DataTypeToString(dst_data_type) << ", ";
    os << "count: " << VectorToString(dims) << ";";
    return os.str();
  };

  auto signature = build_signature();

  auto manager = CastExecutorManager::GetInstance();
  auto gcu_exec = manager->Find(signature);
  if (gcu_exec == nullptr) {
    CompileExecutable(signature, src_data_type, dst_data_type, dims);
  }

  RunExecutableAsync(
      signature, stream, dims, src_data_type, dst_data_type, src_buf, dst_buf);

  VLOG(6) << "end cast runner ";
}

}  // namespace backend
