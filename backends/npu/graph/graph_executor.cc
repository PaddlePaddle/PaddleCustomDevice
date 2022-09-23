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

struct GraphWrapper {
  uint32_t id;
  std::shared_ptr<ge::Graph> ge_graph;
  std::shared_ptr<custom_graph::GEGraph> ctx_graph;
  std::shared_ptr<paddle::framework::ir::IRGraph> ir_graph;
};

static std::unordered_map<C_Scope,
                          std::pair<std::shared_ptr<ge::Session>,
                                    std::unordered_map<C_Graph, GraphWrapper>>>
    session_cache;

bool ge_initialized = false;

C_Status graph_engine_initialize(const C_Device device, const C_Stream stream) {
  if (!ge_initialized) {
    auto soc_name = aclrtGetSocName();
    std::map<ge::AscendString, ge::AscendString> config = {
        {"ge.exec.deviceId", "0"},
        {"ge.graphRunMode", "0"},
        {"ge.exec.precision_mode", "allow_fp32_to_fp16"},
        {"ge.graphMemoryMaxSize", "1048576"},
        {"ge.socVersion", ge::AscendString(soc_name)},
        {"ge.opSelectImplmode", "high_performance"}};
    // {"ge.exec.reuseZeroCopyMemory", "1"},
    // {"GE_USE_STATIC_MEMORY", "2"}
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
  session_cache.clear();
}

C_Status graph_engine_execute_graph(const C_Device device,
                                    const C_Stream stream,
                                    const C_Scope scope,
                                    const C_Graph graph,
                                    char** feed_tensor_name,
                                    void** feed_tensor_data,
                                    size_t feed_tensor_num,
                                    char** fetch_tensor_name,
                                    void** fetch_tensor_data,
                                    size_t fetch_tensor_num) {
  ge::Status ret = ge::SUCCESS;

  graph::utils::log() << "[INFO] scope=" << scope << ", graph=" << graph
                      << std::endl;

  if (session_cache.find(scope) == session_cache.end()) {
    std::map<ge::AscendString, ge::AscendString> options;
    session_cache[scope] = {std::make_shared<ge::Session>(options), {}};
  }
  auto session = session_cache[scope].first.get();

  // 1. Get or create a ge_graph
  bool add_ge_graph = false;
  auto& graph_cache = session_cache[scope].second;
  if (graph_cache.find(graph) == graph_cache.end()) {
    add_ge_graph = true;
    auto id = static_cast<uint32_t>(graph_cache.size());
    std::string name = "ge_graph." + std::to_string(id);
    graph_cache[graph].id = id;
    graph_cache[graph].ge_graph = std::make_shared<ge::Graph>(name);
    graph_cache[graph].ir_graph =
        std::make_shared<paddle::framework::ir::IRGraph>(graph);
    graph_cache[graph].ctx_graph = std::make_shared<custom_graph::GEGraph>(
        name, id, graph_cache[graph].ge_graph.get());
  }

  auto& ir_graph = *graph_cache[graph].ir_graph;
  auto ge_graph_id = graph_cache[graph].id;
  std::string ge_graph_name = "ge_graph." + std::to_string(ge_graph_id);
  auto& global_graph = *graph_cache[graph].ctx_graph;

  if (add_ge_graph) {
    // 2. Get or create variable and constant
    for (auto& var_node : ir_graph.Vars()) {
      // input node
      if (var_node->Persistable() && (var_node->Type() != "feed_minibatch" &&
                                      var_node->Type() != "fetch_list")) {
        std::string var_name = var_node->Name();
        // if (var_node->IsParameter()) {
        auto var_dims = var_node->dims();
        ge::TensorDesc var_desc(
            ge::Shape(std::vector<int64_t>(var_dims.begin(), var_dims.end())),
            ge::Format::FORMAT_NCHW,
            graph::utils::pd_dtype_to_ge_dtype(var_node->dtype()));
        var_desc.SetRealDimCnt(var_desc.GetShape().GetDimNum());

        auto ge_op = ge::op::Variable(ge::AscendString(var_name.c_str()));
        ge_op.update_output_desc_y(var_desc);
        global_graph.AddOp(var_name, ge_op);
        graph::utils::log()
            << "[INFO] var " << var_name
            << ", dims=" << paddle::framework::ir::to_string(var_dims)
            << std::endl;
        // }
      }
    }

    // 3. Build graph
    for (auto& op_node : ir_graph.Ops()) {
      if (custom_graph::OpAdapter::Factory().find(op_node->Type()) !=
          custom_graph::OpAdapter::Factory().end()) {
        graph::utils::log()
            << "[INFO] run " << op_node->Type() << " adapter" << std::endl;
        auto& creator = custom_graph::OpAdapter::Factory()[op_node->Type()];
        auto adaper = creator();
        adaper->run(*op_node, &global_graph);
      } else {
        graph::utils::log() << "[ERROR] op " << op_node->Type()
                            << " is not supported." << std::endl;
        exit(-1);
      }
    }

    if (global_graph.feed_inputs_.size() != feed_tensor_num ||
        global_graph.fetch_outputs_.size() != fetch_tensor_num) {
      graph::utils::log() << "[ERROR] global_graph.feed_inputs_.size(): "
                          << global_graph.feed_inputs_.size()
                          << " != feed_tensor_num: " << feed_tensor_num
                          << " || "
                             "global_graph.fetch_outputs_size(): "
                          << global_graph.fetch_outputs_.size()
                          << " != fetch_tensor_num: " << fetch_tensor_num
                          << std::endl;
      return C_FAILED;
    }

    global_graph.Finalize(session);
  }

  // 4. Run graph
  std::vector<ge::Tensor> input_tensors;
  std::vector<ge::Tensor> output_tensors;

  for (auto i = 0; i < feed_tensor_num; ++i) {
    std::string tensor_name = global_graph.feed_inputs_[i];
    auto var_node = ir_graph.Var(tensor_name);
    auto var_dims = var_node->dims();
    int numel = std::accumulate(
        var_dims.begin(), var_dims.end(), 1, std::multiplies<int>());
    auto tensor_desc = ge::TensorDesc(
        ge::Shape(std::vector<int64_t>(var_dims.begin(), var_dims.end())),
        ge::Format::FORMAT_NCHW,
        graph::utils::pd_dtype_to_ge_dtype(var_node->dtype()));
    void* data_ptr = nullptr;
    for (auto j = 0; j < feed_tensor_num; ++j) {
      if (tensor_name == feed_tensor_name[j]) {
        data_ptr = feed_tensor_data[j];
        break;
      }
    }
    if (!data_ptr) {
      graph::utils::log() << "[ERROR] not found feed_tensor " << tensor_name
                          << std::endl;
      return C_FAILED;
    }
    graph::utils::log() << "[INFO] feed " << tensor_name << ", dims="
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

  ret = session->RunGraph(ge_graph_id, input_tensors, output_tensors);
  if (ret != ge::SUCCESS) {
    graph::utils::log() << "[ERROR] run graph  " << ge_graph_id << ": "
                        << ge_graph_name << " failed." << std::endl;
  } else {
    graph::utils::log() << "[INFO] run graph " << ge_graph_id << ": "
                        << ge_graph_name << " success." << std::endl;
  }

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
    auto& out = output_tensors[i];
    auto out_desc = out.GetTensorDesc();
    auto out_dim = out_desc.GetShape().GetDims();
    auto out_data = out.GetData();
    std::string tensor_name = global_graph.fetch_outputs_[i];
    void* data_ptr = nullptr;
    for (auto j = 0; j < fetch_tensor_num; ++j) {
      if (tensor_name == fetch_tensor_name[j]) {
        data_ptr = fetch_tensor_data[j];
        break;
      }
    }
    if (!data_ptr) {
      graph::utils::log() << "[ERROR] not found fetch_tensor " << tensor_name
                          << std::endl;
      return C_FAILED;
    } else {
      graph::utils::log() << "[INFO] fetch " << tensor_name << ", dims="
                          << paddle::framework::ir::to_string(out_dim)
                          << ", ptr=" << reinterpret_cast<void*>(out_data)
                          << ", size=" << out.GetSize() << std::endl;
      std::memcpy(data_ptr, out_data, out.GetSize());
    }
  }

  return C_SUCCESS;
}
