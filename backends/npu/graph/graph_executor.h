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

#include "graph/graph_funcs.h"
#include "graph/graph_utils.h"
#include "graph/paddle_graph.h"

// NOLINT
#include "all_ops.h"
#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "ge/ge_error_codes.h"
#include "ge/ge_ir_build.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "graph/types.h"

namespace custom_graph {

template <typename T = int>
inline void UpdatePaddingAndDilation(std::vector<T>* paddings,
                                     std::vector<T>* dilation,
                                     const std::string padding_algorithm,
                                     const std::vector<T> data_dims,
                                     const std::vector<T>& strides,
                                     const std::vector<T>& ksize) {
  // set padding size == data_dims.size() * 2
  auto data_shape = data_dims;
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    if (data_dims.size() * 2 != paddings->size()) {
      // error
    }
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + ksize[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;

      // dilation
      *(dilation->begin() + i) = 1;
    }

  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

template <typename T = int>
inline void UpdatePadding(std::vector<T>* paddings,
                          const bool global_pooling,
                          const bool adaptive,
                          const std::string padding_algorithm,
                          const std::vector<T> data_dims,
                          const std::vector<T>& strides,
                          const std::vector<T>& kernel_size) {
  // set padding size == data_dims.size() * 2
  auto data_shape = data_dims;
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    // PADDLE_ENFORCE_EQ(data_dims.size() * 2,
    //                   paddings->size(),
    //                   errors::InvalidArgument(
    //                       "Paddings size %d should be the same or twice as
    //                       the " "pooling size %d.", paddings->size(),
    //                       data_dims.size() * 2));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + kernel_size[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }

  // if global_pooling == true or adaptive == true, padding will be ignore
  if (global_pooling || adaptive) {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

template <typename T>
inline std::vector<T> slice_ddim(const std::vector<T>& dim,
                                 int begin,
                                 int end) {
  return std::vector<T>(dim.cbegin() + begin, dim.cbegin() + end);
}

class GEGraph {
 public:
  GEGraph(const std::string& ge_graph_name,
          uint32_t ge_graph_id,
          ge::Graph* ge_graph)
      : ge_graph_name_(ge_graph_name),
        ge_graph_id_(ge_graph_id),
        ge_graph_(ge_graph) {}

  void AddInput(ge::Operator in) { inputs_.push_back(in); }

  void AddFeedInput(std::string name, ge::Operator in, int col) {
    if (feed_inputs_.size() <= col) {
      feed_inputs_.resize(col + 1);
    }
    feed_inputs_[col] = name;
    AddInput(in);
  }

  void AddOutput(ge::Operator out) { outputs_.push_back(out); }

  void AddFetchOutput(std::string name, ge::Operator out) {
    fetch_outputs_.push_back(name);
    std::cout << fetch_outputs_.size() << std::endl;
    AddOutput(out);
  }

  ge::Operator& GetOp(const std::string& op) {
    if (ge_ops_.find(op) == ge_ops_.end()) {
      std::cerr << "not found " << op << " in context " << this << std::endl;
      exit(-1);
    } else {
      std::cout << "found " << op << " in context " << this << std::endl;
    }
    return ge_ops_[op];
  }

  bool HasOp(const std::string& op) {
    return ge_ops_.find(op) != ge_ops_.end();
  }

  ge::Operator& AddOp(const std::string& op, ge::Operator ge_op) {
    RecordNode(op, ge_op);
    ge_graph_->AddOp(ge_op);
    return ge_ops_[op];
  }

  ge::Operator& RecordNode(const std::string& op, ge::Operator ge_op) {
    std::cout << "record " << op << " in context " << this << std::endl;
    ge_ops_[op] = ge_op;
  }

  ge::Graph* Graph() { return ge_graph_; }

  const std::string& GraphName() { return ge_graph_name_; }

  uint32_t GraphId() { return ge_graph_id_; }

  void Finalize(ge::Session* session) {
    ge_graph_->SetInputs(inputs_);
    ge_graph_->SetOutputs(outputs_);

    auto ret = ge::aclgrphDumpGraph(
        *ge_graph_, ge_graph_name_.c_str(), ge_graph_name_.size());
    if (ret != ge::SUCCESS) {
      std::cout << "Save graph  " << ge_graph_id_ << ": " << ge_graph_name_
                << " failed.\n";
    } else {
      std::cout << "Save graph " << ge_graph_id_ << ": " << ge_graph_name_
                << " success.\n";
    }
    ret = session->AddGraph(ge_graph_id_, *ge_graph_);
    if (ret != ge::SUCCESS) {
      std::cout << "Add graph  " << ge_graph_id_ << ": " << ge_graph_name_
                << " failed.\n";
    } else {
      std::cout << "Add graph " << ge_graph_id_ << ": " << ge_graph_name_
                << " success.\n";
    }
  }

  //  private:
  std::string ge_graph_name_;
  uint32_t ge_graph_id_;
  ge::Graph* ge_graph_;  // not own

  std::unordered_map<std::string, ge::Operator> ge_ops_{};
  std::vector<ge::Operator> inputs_{};
  std::vector<ge::Operator> outputs_{};
  std::vector<std::string> feed_inputs_{};
  std::vector<std::string> fetch_outputs_{};
};

class OpAdapter;

using adapter_creator_t = std::function<std::shared_ptr<OpAdapter>()>;

class OpAdapter {
 public:
  static std::unordered_map<std::string, adapter_creator_t>& Factory() {
    static std::unordered_map<std::string, adapter_creator_t> factory;
    return factory;
  }

  OpAdapter() = default;

  OpAdapter& self() { return *this; }

  virtual ~OpAdapter() {}

  virtual void run(const paddle::framework::ir::OpNode& ctx,
                   GEGraph* graph) = 0;
};

template <typename AdapterT>
class Registrar {
 public:
  Registrar(const std::string& ir) {
    adapter_creator_t adapter_creator = []() -> std::shared_ptr<OpAdapter> {
      return std::make_shared<AdapterT>();
    };
    OpAdapter::Factory()[ir] = std::move(adapter_creator);
  }

  int Touch() { return 0; }
};

#define REG_OP_ADAPTER(ir, adapter)                                          \
  static ::custom_graph::Registrar<adapter> __op_adapter_registrar_##ir##__( \
      #ir);                                                                  \
  int __op_adapter_registrar_##ir##__touch__() {                             \
    return __op_adapter_registrar_##ir##__.Touch();                          \
  }

#define USE_OP_ADAPTER(ir)                                              \
  extern int __op_adapter_registrar_##ir##__touch__();                  \
  static __attribute__((unused)) int __use_op_adapter_##ir##__touch__ = \
      __op_adapter_registrar_##ir##__touch__()

}  // namespace custom_graph
