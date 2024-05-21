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

#ifdef PADDLE_WITH_ATB
#include "runner.h"  // NOLINT

DECLARE_bool(npu_runtime_debug);
DECLARE_bool(npu_blocking_run);

namespace atb_layers {

// TaskQueue
TaskQueue& TaskQueue::Instance(int dev_id) {
  static std::unordered_map<int, std::unique_ptr<TaskQueue>> ins;
  if (ins.find(dev_id) == ins.end()) {
    ins[dev_id] = std::make_unique<TaskQueue>(dev_id);
  }
  return *ins[dev_id];
}

TaskQueue::TaskQueue(int dev_id) {
  is_running_ = true;
  thread_ = std::move(std::thread([this, dev_id] {
    C_Device_st device{dev_id};
    SetDevice(&device);
    while (this->is_running_) {
      {
        std::unique_lock<std::mutex> lock(this->mutex_);
        this->cond_.wait(lock, [this] {
          return !this->is_running_ || !this->task_list_.empty();
        });
        while (!this->task_list_.empty()) {
          auto task = std::move(this->task_list_.front());
          task();
          this->task_list_.pop_front();
        }
      }
    }
  }));
}

TaskQueue::~TaskQueue() { Stop(); }

void TaskQueue::Stop() {
  if (is_running_) {
    is_running_ = false;
    cond_.notify_all();
    thread_.join();
  }
}

void TaskQueue::Commit(std::packaged_task<void(void)>&& task) {
  if (is_running_) {
    if (FLAGS_npu_blocking_run) {
      task();
    } else {
      std::unique_lock<std::mutex> lock(mutex_);
      task_list_.emplace_back(std::move(task));
      lock.unlock();
      cond_.notify_all();
    }
  }
  LOG_IF(INFO, FLAGS_npu_runtime_debug)
      << "[RUNTIME] Commit Task to TaskQueue. Id=" << task_id_++;
}

void TaskQueue::Wait() {
  std::packaged_task<void(void)> wait_task([this]() {
    LOG_IF(INFO, FLAGS_npu_runtime_debug)
        << "[RUNTIME] Wait TaskQueue. Id=" << this->task_id_;
  });
  auto wait_task_future = wait_task.get_future();
  Commit(std::move(wait_task));
  wait_task_future.get();
}

// OperationRunner
void OperationRunner::reset_variant_pack() {
  variant_pack_.inTensors.resize(0);
  variant_pack_.outTensors.resize(0);
  in_tensor_num_ = 0;
  out_tensor_num_ = 0;
  workspace_ = nullptr;
  workspace_size_ = 0;
}

void OperationRunner::bind_input(const void* src,
                                 const phi::DataType& dtype,
                                 const std::vector<int64_t>& dims) {
  variant_pack_.inTensors.resize(in_tensor_num_ + 1);
  atb::Tensor& atb_tensor = variant_pack_.inTensors[in_tensor_num_++];
  atb_tensor.desc.format = ACL_FORMAT_ND;
  atb_tensor.desc.shape.dimNum = dims.size();
  for (auto i = 0; i < dims.size(); ++i) {
    atb_tensor.desc.shape.dims[i] = dims[i];
  }
  atb_tensor.desc.dtype = ConvertToNpuDtype(dtype);
  atb_tensor.dataSize = atb::Utils::GetTensorSize(atb_tensor);
  atb_tensor.hostData = nullptr;
  atb_tensor.deviceData = const_cast<void*>(src);
}

void OperationRunner::bind_host_input(const void* src,
                                      const phi::DataType& dtype,
                                      const std::vector<int64_t>& dims) {
  variant_pack_.inTensors.resize(in_tensor_num_ + 1);
  atb::Tensor& atb_tensor = variant_pack_.inTensors[in_tensor_num_++];
  atb_tensor.desc.format = ACL_FORMAT_ND;
  atb_tensor.desc.shape.dimNum = dims.size();
  for (auto i = 0; i < dims.size(); ++i) {
    atb_tensor.desc.shape.dims[i] = dims[i];
  }
  atb_tensor.desc.dtype = ConvertToNpuDtype(dtype);
  atb_tensor.dataSize = atb::Utils::GetTensorSize(atb_tensor);
  atb_tensor.deviceData = nullptr;
  atb_tensor.hostData = const_cast<void*>(src);
}

void OperationRunner::bind_input(const void* src,
                                 const void* host_src,
                                 const phi::DataType& dtype,
                                 const std::vector<int64_t>& dims) {
  variant_pack_.inTensors.resize(in_tensor_num_ + 1);
  atb::Tensor& atb_tensor = variant_pack_.inTensors[in_tensor_num_++];
  atb_tensor.desc.format = ACL_FORMAT_ND;
  atb_tensor.desc.shape.dimNum = dims.size();
  for (auto i = 0; i < dims.size(); ++i) {
    atb_tensor.desc.shape.dims[i] = dims[i];
  }
  atb_tensor.desc.dtype = ConvertToNpuDtype(dtype);
  atb_tensor.dataSize = atb::Utils::GetTensorSize(atb_tensor);
  atb_tensor.hostData = const_cast<void*>(host_src);
  atb_tensor.deviceData = const_cast<void*>(src);
}

void OperationRunner::bind_input(const phi::DenseTensor& src,
                                 const std::vector<int64_t>& dims) {
  bool is_cpu_tensor = src.place().GetType() == phi::AllocationType::CPU;
  std::vector<int64_t> new_dims;
  if (dims.size() == 0) {
    new_dims = phi::vectorize<int64_t>(src.dims());
  } else {
    new_dims = dims;
  }
  if (is_cpu_tensor) {
    bind_host_input(src.data(), src.dtype(), new_dims);
  } else {
    bind_input(src.data(), src.dtype(), new_dims);
  }
}

void OperationRunner::bind_input(const phi::DenseTensor& src,
                                 const phi::DenseTensor& host_src,
                                 const std::vector<int64_t>& dims) {
  std::vector<int64_t> new_dims;
  if (dims.size() == 0) {
    new_dims = phi::vectorize<int64_t>(src.dims());
  } else {
    new_dims = dims;
  }
  bind_input(src.data(), host_src.data(), src.dtype(), new_dims);
}

void OperationRunner::bind_output(const void* src,
                                  const phi::DataType& dtype,
                                  const std::vector<int64_t>& dims) {
  variant_pack_.outTensors.resize(out_tensor_num_ + 1);
  atb::Tensor& atb_tensor = variant_pack_.outTensors[out_tensor_num_++];
  atb_tensor.desc.format = ACL_FORMAT_ND;
  atb_tensor.desc.shape.dimNum = dims.size();
  for (auto i = 0; i < dims.size(); ++i) {
    atb_tensor.desc.shape.dims[i] = dims[i];
  }
  atb_tensor.desc.dtype = ConvertToNpuDtype(dtype);
  atb_tensor.dataSize = atb::Utils::GetTensorSize(atb_tensor);
  atb_tensor.hostData = nullptr;
  atb_tensor.deviceData = const_cast<void*>(src);
}

void OperationRunner::bind_output(phi::DenseTensor* src,
                                  const std::vector<int64_t>& dims) {
  std::vector<int64_t> new_dims;
  if (dims.size() == 0) {
    new_dims = phi::vectorize<int64_t>(src->dims());
  } else {
    new_dims = dims;
  }
  bind_output(src->data(), src->dtype(), new_dims);
}

void OperationRunner::setup(const phi::CustomContext& dev_ctx) {
  auto ctx = get_context(dev_ctx);
  ATB_CHECK(operation_->Setup(variant_pack_, workspace_size_, ctx));
  if (workspace_size_ > 0) {
    workspace_ = get_workspace(dev_ctx, workspace_size_);
  }
}

void OperationRunner::execute(const phi::CustomContext& dev_ctx) {
  auto ctx = get_context(dev_ctx);
  ATB_CHECK(
      operation_->Execute(variant_pack_, workspace_, workspace_size_, ctx));
  if (FLAGS_npu_blocking_run) {
    dev_ctx.Wait();
  }
}

void OperationRunner::run(const phi::CustomContext& dev_ctx) {
  setup(dev_ctx);
  execute(dev_ctx);
}

atb::Context* OperationRunner::get_context(const phi::CustomContext& dev_ctx) {
  static std::unordered_map<void*, atb::Context*> m;

  void* stream =
      const_cast<void*>(reinterpret_cast<const void*>(dev_ctx.stream()));
  if (m.find(stream) == m.end()) {
    ATB_CHECK(atb::CreateContext(&m[stream]));
    ATB_CHECK(m[stream]->SetExecuteStream(reinterpret_cast<void*>(stream)));
  }
  return m[stream];
}

uint8_t* OperationRunner::get_workspace(const phi::CustomContext& dev_ctx,
                                        uint64_t workspace_size) {
  static phi::DenseTensor tmp;
  if (workspace_size > tmp.numel()) {
    dev_ctx.Wait();
    tmp.Resize({workspace_size});
    dev_ctx.template Alloc<uint8_t>(&tmp);
    LOG(INFO) << "alloc workspace size: " << tmp.numel();
  }
  if (tmp.numel() == 0) {
    return nullptr;
  }
  return tmp.data<uint8_t>();
}

}  // namespace atb_layers

#endif
