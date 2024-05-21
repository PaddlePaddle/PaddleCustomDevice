// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_ATB

#include "atb/atb_infer.h"
#include "kernels/funcs/npu_funcs.h"

#define ATB_CHECK(func) RUNTIME_CHECK(func, atb::NO_ERROR)

namespace atb_layers {

template <typename T>
struct to_acl_dtype;

template <>
struct to_acl_dtype<int8_t> {
  static const aclDataType value() { return ACL_INT8; }
};

class TaskQueue {
 public:
  static TaskQueue& Instance(int dev_id);

  uint64_t GetTaskId() { return task_id_; }

  explicit TaskQueue(int dev_id);

  ~TaskQueue();

  void Stop();

  void Commit(std::packaged_task<void(void)>&& task);

  void Wait();

 private:
  std::mutex mutex_;
  std::condition_variable cond_;
  std::list<std::packaged_task<void(void)>> task_list_;
  std::thread thread_;
  uint64_t task_id_{0};
  bool is_running_{true};
};

class OperationRunner {
 public:
  OperationRunner() {}

  bool is_initialized() const { return operation_ != nullptr; }

  template <typename OpParam>
  void create(const OpParam& param) {
    ATB_CHECK(atb::CreateOperation(param, &operation_));
  }

  void reset_variant_pack();

  void bind_input(const void* src,
                  const phi::DataType& dtype,
                  const std::vector<int64_t>& dims);

  void bind_host_input(const void* src,
                       const phi::DataType& dtype,
                       const std::vector<int64_t>& dims);

  void bind_input(const void* src,
                  const void* host_src,
                  const phi::DataType& dtype,
                  const std::vector<int64_t>& dims);

  void bind_input(const phi::DenseTensor& src,
                  const std::vector<int64_t>& dims = {});

  template <typename T>
  void bind_input() {
    variant_pack_.inTensors.resize(in_tensor_num_ + 1);
    atb::Tensor& atb_tensor = variant_pack_.inTensors[in_tensor_num_++];
    atb_tensor.dataSize = 0;
    atb_tensor.hostData = nullptr;
    atb_tensor.deviceData = nullptr;
    atb_tensor.desc.format = ACL_FORMAT_ND;
    atb_tensor.desc.dtype = to_acl_dtype<T>::value();
  }

  void bind_input(const phi::DenseTensor& src,
                  const phi::DenseTensor& host_src,
                  const std::vector<int64_t>& dims = {});

  void bind_output(const void* src,
                   const phi::DataType& dtype,
                   const std::vector<int64_t>& dims);

  void bind_output(phi::DenseTensor* src,
                   const std::vector<int64_t>& dims = {});

  void bind_input(const paddle::Tensor& src,
                  const std::vector<int64_t>& dims = {}) {
    bind_input(*static_cast<const phi::DenseTensor*>(src.impl().get()), dims);
  }

  void bind_input(const paddle::Tensor& src,
                  const paddle::Tensor& host_src,
                  const std::vector<int64_t>& dims = {}) {
    bind_input(*static_cast<const phi::DenseTensor*>(src.impl().get()),
               *static_cast<const phi::DenseTensor*>(host_src.impl().get()),
               dims);
  }

  void bind_output(paddle::Tensor* src, const std::vector<int64_t>& dims = {}) {
    bind_output(static_cast<phi::DenseTensor*>(src->impl().get()), dims);
  }

  void setup(const phi::CustomContext& dev_ctx);

  void execute(const phi::CustomContext& dev_ctx);

  void run(const phi::CustomContext& dev_ctx);

 private:
  atb::Context* get_context(const phi::CustomContext& dev_ctx);
  uint8_t* get_workspace(const phi::CustomContext& dev_ctx,
                         uint64_t workspace_size);

 private:
  atb::Operation* operation_ = nullptr;
  atb::VariantPack variant_pack_;
  uint64_t in_tensor_num_ = 0;
  uint64_t out_tensor_num_ = 0;
  uint8_t* workspace_ = nullptr;
  uint64_t workspace_size_ = 0;
};

}  // namespace atb_layers

#endif
