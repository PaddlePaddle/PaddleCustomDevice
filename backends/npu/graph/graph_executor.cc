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

void update_op_format(ge::Operator op, ge::Format format) {
  ge::TensorDesc tensor_desc_x = op.GetInputDescByName("x");
  ge::TensorDesc tensor_desc_y = op.GetInputDescByName("y");
  tensor_desc_x.SetFormat(format);
  tensor_desc_y.SetFormat(format);
  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateOutputDesc("y", tensor_desc_y);
}

// static std::shared_ptr<ge::Session> session; // process hang, if no delete it
ge::Session* session = nullptr;
static std::unordered_map<C_Graph,
                          std::pair<uint32_t, std::shared_ptr<ge::Graph>>>
    ge_graph_map;
static std::unordered_map<C_Graph,
                          std::shared_ptr<paddle::framework::ir::IRGraph>>
    ir_graph_map;
static std::unordered_map<C_Graph, std::shared_ptr<custom_graph::GEGraph>>
    ctx_graph_map;

bool ge_initialized = false;

C_Status graph_engine_initialize(const C_Device device, const C_Stream stream) {
  if (!ge_initialized) {
    std::map<ge::AscendString, ge::AscendString> config = {
        {"ge.exec.deviceId", "0"},
        {"ge.graphRunMode", "1"},
        {"ge.exec.precision_mode", "allow_fp32_to_fp16"}};
    ge::Status ret = ge::GEInitialize(config);
    if (ret != ge::SUCCESS) {
      std::cout << "Initialize ge failed.\n";
      return C_FAILED;
    }
    std::cout << "Initialize ge success.\n";
  }

  // session = std::make_shared<ge::Session>(options);
  if (!session) {
    std::map<ge::AscendString, ge::AscendString> options;
    session = new ge::Session(options);
  }

  // session->RegisterCallBackFunc("Save", CallBack);
  return C_SUCCESS;
}

C_Status graph_engine_finalize(const C_Device device, const C_Stream stream) {
  if (session) {
    delete session;
    session = nullptr;
  }
}

C_Status graph_engine_prepare(const C_Device device,
                              const C_Stream stream,
                              const C_Graph graph,
                              char** init_tensor_name,
                              void** init_tensor_data,
                              size_t init_tensor_num) {}

C_Status graph_engine_execute_graph(const C_Device device,
                                    const C_Stream stream,
                                    const C_Graph graph,
                                    char** feed_tensor_name,
                                    void** feed_tensor_data,
                                    size_t feed_tensor_num,
                                    char** fetch_tensor_name,
                                    void** fetch_tensor_data,
                                    size_t fetch_tensor_num) {
  ge::Status ret = ge::SUCCESS;

  // 1. Get or create a ge_graph
  bool add_ge_graph = false;
  if (ge_graph_map.find(graph) == ge_graph_map.end()) {
    uint32_t id = static_cast<uint32_t>(ge_graph_map.size());
    std::string name = "ge_graph." + std::to_string(id);
    ge_graph_map[graph] = {id, std::make_shared<ge::Graph>(name)};
    ir_graph_map[graph] =
        std::make_shared<paddle::framework::ir::IRGraph>(graph);
    ctx_graph_map[graph] = std::make_shared<custom_graph::GEGraph>(
        name, id, ge_graph_map[graph].second.get());

    add_ge_graph = true;
  }
  auto& ir_graph = *ir_graph_map[graph];
  uint32_t ge_graph_id = ge_graph_map[graph].first;
  std::string ge_graph_name = "ge_graph." + std::to_string(ge_graph_id);
  auto& ge_graph = ge_graph_map[graph].second;

  custom_graph::GEGraph& global_graph = *ctx_graph_map[graph];

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
        std::cout << "var " << var_name
                  << ", dims=" << paddle::framework::ir::to_string(var_dims)
                  << std::endl;
        // }
      }
    }

    // 3. Build graph
    for (auto& op_node : ir_graph.Ops()) {
      if (custom_graph::OpAdapter::Factory().find(op_node->Type()) !=
          custom_graph::OpAdapter::Factory().end()) {
        std::cout << "run " << op_node->Type() << " adapter\n";
        auto& creator = custom_graph::OpAdapter::Factory()[op_node->Type()];
        auto adaper = creator();
        adaper->run(*op_node, &global_graph);
      } else {
        std::cerr << "unsupported op " << op_node->Type() << std::endl;
        exit(-1);
      }
    }

    if (global_graph.feed_inputs_.size() != feed_tensor_num ||
        global_graph.fetch_outputs_.size() != fetch_tensor_num) {
      std::cerr << "global_graph.feed_inputs_.size(): "
                << global_graph.feed_inputs_.size()
                << " != feed_tensor_num: " << feed_tensor_num
                << " || "
                   "global_graph.fetch_outputs_size(): "
                << global_graph.fetch_outputs_.size()
                << " != fetch_tensor_num: " << fetch_tensor_num << "\n";
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
      std::cerr << "not found feed_tensor " << tensor_name << std::endl;
      return C_FAILED;
    }
    // std::cout << "feed tensor_name: " << tensor_name << ", ptr=" << data_ptr
    //           << ", size="
    //           << numel * graph::utils::get_pd_dtype_size(var_node->dtype())
    //           << std::endl;
    input_tensors.push_back(
        ge::Tensor(tensor_desc,
                   reinterpret_cast<uint8_t*>(data_ptr),
                   numel * graph::utils::get_pd_dtype_size(var_node->dtype())));
  }

  ret = session->RunGraph(ge_graph_id, input_tensors, output_tensors);
  if (ret != ge::SUCCESS) {
    std::cout << "Run graph  " << ge_graph_id << ": " << ge_graph_name
              << " failed.\n";
  } else {
    std::cout << "Run graph " << ge_graph_id << ": " << ge_graph_name
              << " success.\n";
  }

  if (output_tensors.size() != fetch_tensor_num) {
    std::cerr << "output_tensors.size(): " << output_tensors.size()
              << " != fetch_tensor_num: " << fetch_tensor_num << "\n";
    return C_FAILED;
  }

  std::cout << "output_tensors size " << output_tensors.size()
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
      std::cerr << "not found fetch_tensor " << tensor_name << std::endl;
      return C_FAILED;
    }
    std::memcpy(data_ptr, out_data, out.GetSize());
  }

  return C_SUCCESS;
}
