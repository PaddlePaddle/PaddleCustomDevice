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

bool ge_initialized = false;
ge::Session* session = nullptr;
std::unordered_map<C_Graph, std::shared_ptr<custom_graph::Graph>> graph_cache;

std::map<ge::AscendString, ge::AscendString> config;

C_Status graph_engine_initialize(const C_Device device, const C_Stream stream) {
  if (!ge_initialized) {
    ge_initialized = true;
    auto soc_name = aclrtGetSocName();
    config = {{"ge.exec.deviceId",
               ge::AscendString(std::to_string(device->id).c_str())},
              {"ge.graphRunMode", "0"},
              {"ge.exec.precision_mode", "allow_fp32_to_fp16"},
              {"ge.graphMemoryMaxSize", "22548578304"},
              {"ge.variableMemoryMaxSize",
               "10737418240"}, /* graphMemoryMaxSize + variableMemoryMaxSize <=
                                  31 GB */
              {"ge.socVersion", ge::AscendString(soc_name)},
              {"ge.opSelectImplmode", "high_performance"}};
    // {"ge.exec.reuseZeroCopyMemory", "1"},
    // {"GE_USE_STATIC_MEMORY", "2"}};
    ge::Status ret = ge::GEInitialize(config);
    if (ret != ge::SUCCESS) {
      graph::utils::log() << "[ERROR] graph_engine_initialize failed."
                          << std::endl;
      return C_FAILED;
    }
    graph::utils::log() << "[INFO] graph_engine_initialize success."
                        << std::endl;
  }
  return C_SUCCESS;
}

C_Status graph_engine_finalize(const C_Device device, const C_Stream stream) {
  if (ge_initialized) {
    ge_initialized = false;
    // for (auto& iter : session_cache) {
    //   iter.second.second.clear();
    // }
    // session_cache.clear();
    if (session) {
      graph_cache.clear();
      custom_graph::Tensor::TensorStorage().clear();
      delete session;
      session = nullptr;
    }
    ge::Status ret = ge::GEFinalize();
    if (ret != ge::SUCCESS) {
      graph::utils::log() << "[ERROR] graph_engine_finalize failed."
                          << std::endl;
      return C_FAILED;
    }
    graph::utils::log() << "[INFO] graph_engine_finalize success." << std::endl;
  }
  return C_SUCCESS;
}

C_Status graph_engine_execute_graph(const C_Device device,
                                    const C_Stream stream,
                                    const C_Scope c_scope,
                                    const C_Graph c_graph,
                                    char** feed_tensor_name,
                                    void** feed_tensor_data,
                                    size_t feed_tensor_num,
                                    char** fetch_tensor_name,
                                    void** fetch_tensor_data,
                                    size_t fetch_tensor_num) {
  graph::utils::log() << "[INFO] run graph_engine_execute_graph" << std::endl;

  if (session == nullptr) {
    session = new ge::Session(config);
  }

  bool add_ge_graph = false;
  if (graph_cache.find(c_graph) == graph_cache.end()) {
    auto graph_id = graph_cache.size();
    graph_cache.insert(
        {c_graph,
         std::make_shared<custom_graph::Graph>(
             "graph_" + std::to_string(graph_id), graph_id, c_graph)});
    add_ge_graph = true;
  }
  auto graph = graph_cache[c_graph];

  auto paddle_ir_graph = graph->ir_graph();
  custom_graph::Graph::set_global_graph(graph.get());

  // 1. build graph
  if (add_ge_graph) {
    // 1.1 build vars
    for (auto& var_node : paddle_ir_graph->Vars()) {
      if (var_node->Persistable() && (var_node->Type() != "feed_minibatch" &&
                                      var_node->Type() != "fetch_list")) {
        auto& tensor = custom_graph::Tensor::Get(var_node->Name())
                           .SetShape(var_node->dims())
                           .SetDType(var_node->dtype())
                           .SetFormat("NCHW");
        custom_graph::OpCommand("Variable", var_node->Name())
            .Output(tensor, "y");
        graph::utils::log()
            << "[INFO] var " << var_node->Name()
            << ", dims=" << paddle::framework::ir::to_string(var_node->dims())
            << std::endl;
      }
    }

    // 1.2 build ops
    for (auto& op_node : paddle_ir_graph->Ops()) {
      if (custom_graph::OpAdapter::Factory().find(op_node->Type()) !=
          custom_graph::OpAdapter::Factory().end()) {
        graph::utils::log()
            << "[INFO] run " << op_node->Type() << " adapter" << std::endl;
        auto& creator = custom_graph::OpAdapter::Factory()[op_node->Type()];
        auto adapter = creator();
        custom_graph::Context adapter_context(op_node.get());
        adapter->run(adapter_context);
      } else {
        graph::utils::log() << "[ERROR] op " << op_node->Type()
                            << " is not supported." << std::endl;
        return C_FAILED;
      }
    }

    // 1.3 finalize
    std::vector<ge::Operator> inputs;
    std::vector<ge::Operator> outputs;
    std::vector<ge::Operator> targets;
    for (auto& op : graph->GetInputs()) {
      inputs.push_back(*op->GetGEOp());
    }
    for (auto& op : graph->GetOutputs()) {
      outputs.push_back(*op->GetGEOp());
    }
    for (auto& op : graph->GetTargets()) {
      outputs.push_back(*op->GetGEOp());
    }
    graph->ge_graph()->SetInputs(inputs);
    graph->ge_graph()->SetOutputs(outputs);
    graph->ge_graph()->SetTargets(targets);

    if (ge::aclgrphDumpGraph(*graph->ge_graph(),
                             graph->GetName().c_str(),
                             graph->GetName().size()) != ge::SUCCESS) {
      graph::utils::log() << "[ERROR] save graph  " << graph->GetId() << ": "
                          << graph->GetName() << " failed." << std::endl;
    } else {
      graph::utils::log() << "[INFO] save graph " << graph->GetId() << ": "
                          << graph->GetName() << " success." << std::endl;
    }

    if (session->AddGraph(graph->GetId(), *graph->ge_graph()) != ge::SUCCESS) {
      graph::utils::log() << "[ERROR] add graph  " << graph->GetId() << ": "
                          << graph->GetName() << " failed." << std::endl;
    } else {
      graph::utils::log() << "[INFO] add graph " << graph->GetId() << ": "
                          << graph->GetName() << " success." << std::endl;
    }
  }

  // 2. Run graph
  std::vector<ge::Tensor> input_tensors;
  std::vector<ge::Tensor> output_tensors;

  std::vector<std::string> feed_inputs;
  std::vector<std::string> fetch_outputs;
  for (auto& tensor : graph->GetFeedInputs()) {
    feed_inputs.push_back(tensor->Name());
  }
  for (auto& tensor : graph->GetFetchOutputs()) {
    fetch_outputs.push_back(tensor->Name());
  }

  // 2.1 feed data
  for (auto i = 0; i < feed_tensor_num; ++i) {
    std::string name = feed_inputs[i];
    auto var_node = paddle_ir_graph->Var(name);
    auto var_dims = var_node->dims();
    int numel = std::accumulate(
        var_dims.begin(), var_dims.end(), 1, std::multiplies<int>());
    auto tensor_desc = ge::TensorDesc(
        ge::Shape(std::vector<int64_t>(var_dims.begin(), var_dims.end())),
        ge::Format::FORMAT_NCHW,
        graph::utils::pd_dtype_to_ge_dtype(var_node->dtype()));
    void* data_ptr = nullptr;
    for (auto j = 0; j < feed_tensor_num; ++j) {
      if (name == feed_tensor_name[j]) {
        data_ptr = feed_tensor_data[j];
        break;
      }
    }
    if (!data_ptr) {
      graph::utils::log() << "[ERROR] not found feed_tensor " << name
                          << std::endl;
      return C_FAILED;
    }
    graph::utils::log() << "[INFO] feed " << name << ", dims="
                        << paddle::framework::ir::to_string(var_dims)
                        << ", ptr=" << data_ptr << ", size="
                        << numel * graph::utils::get_pd_dtype_size(
                                       var_node->dtype())
                        << std::endl;
    input_tensors.push_back(
        ge::Tensor(tensor_desc,
                   reinterpret_cast<uint8_t*>(data_ptr),
                   numel * graph::utils::get_pd_dtype_size(var_node->dtype())));
  }

  // 2.2 run graph
  if (session->RunGraph(graph->GetId(), input_tensors, output_tensors) !=
      ge::SUCCESS) {
    graph::utils::log() << "[ERROR] run graph  " << graph->GetId() << ": "
                        << graph->GetName() << " failed." << std::endl;
  } else {
    graph::utils::log() << "[INFO] run graph " << graph->GetId() << ": "
                        << graph->GetName() << " success." << std::endl;
  }

  // 2.3 fetch data
  if (output_tensors.size() != fetch_tensor_num) {
    graph::utils::log() << "[ERROR] output_tensors.size(): "
                        << output_tensors.size()
                        << " != fetch_tensor_num: " << fetch_tensor_num
                        << std::endl;
    return C_FAILED;
  }

  graph::utils::log() << "[INFO] output_tensors size " << output_tensors.size()
                      << ", fetch size " << fetch_tensor_num << std::endl;
  for (auto i = 0; i < output_tensors.size(); ++i) {
    std::string name = fetch_outputs[i];
    auto& out = output_tensors[i];
    auto out_desc = out.GetTensorDesc();
    auto out_dim = out_desc.GetShape().GetDims();
    auto out_data = out.GetData();
    void* data_ptr = nullptr;
    for (auto j = 0; j < fetch_tensor_num; ++j) {
      if (name == fetch_tensor_name[j]) {
        data_ptr = fetch_tensor_data[j];
        break;
      }
    }
    if (!data_ptr) {
      graph::utils::log() << "[ERROR] not found fetch_tensor " << name
                          << std::endl;
      return C_FAILED;
    } else {
      graph::utils::log() << "[INFO] fetch " << name << ", dims="
                          << paddle::framework::ir::to_string(out_dim)
                          << ", ptr=" << reinterpret_cast<void*>(out_data)
                          << ", size=" << out.GetSize() << std::endl;
      std::memcpy(data_ptr, out_data, out.GetSize());
    }
  }

  return C_SUCCESS;
}
