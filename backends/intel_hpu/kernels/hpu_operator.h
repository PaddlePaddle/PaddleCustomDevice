// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BACKENDS_INTEL_HPU_KERNELS_HPU_OPERATOR_H_  // NOLINT
#define BACKENDS_INTEL_HPU_KERNELS_HPU_OPERATOR_H_  // NOLINT

#include <assert.h>

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "glog/logging.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/extension.h"

#define ENABLE_ASYNC_RUN

class HpuOperator {
 public:
  explicit HpuOperator(const std::string guid, bool is_eager = true)
      : guid_(guid), is_eager_(is_eager) {
    if (is_eager_) {
      synStatus status = synGraphCreateEager(&graphHandle_, synDeviceGaudi2);
      PD_CHECK(status == synSuccess,
               "synGraphCreateEager() ",
               guid_,
               " failed = ",
               status);
    } else {
      synStatus status = synGraphCreate(&graphHandle_, synDeviceGaudi2);
      PD_CHECK(status == synSuccess,
               "synGraphCreate() ",
               guid_,
               " failed = ",
               status);
    }
  }

  void Compile();
  virtual ~HpuOperator() {}
  synSectionHandle createSection();
  synTensor createTensor(unsigned dims,
                         synDataType data_type,
                         DIMS tensor_size,
                         bool is_presist,
                         std::string name,
                         synSectionHandle section = nullptr);

 public:
  synRecipeHandle GetRecipe() { return recipeHandle_; }

 protected:
  std::string guid_;
  synGraphHandle graphHandle_;
  synRecipeHandle recipeHandle_;
  std::vector<synSectionHandle> sectons_;
  bool is_eager_;

  std::map<std::string, synTensor> tensors_;
};

#ifdef ENABLE_ASYNC_RUN
class GlobalWorkStreamExecutor {
 public:
  static GlobalWorkStreamExecutor& instance() {
    static GlobalWorkStreamExecutor executor;
    return executor;
  }

  template <typename R>
  std::future<R> async(std::function<R()> func) {
    auto task = std::make_shared<std::packaged_task<R()>>(std::move(func));
    std::future<R> res = task->get_future();
    add_task([task]() { (*task)(); });
    return res;
  }

  template <typename F>
  auto async(F&& func) -> std::future<decltype(func())> {
    using R = decltype(func());
    auto task =
        std::make_shared<std::packaged_task<R()>>(std::forward<F>(func));
    std::future<R> res = task->get_future();
    add_task([task]() { (*task)(); });
    return res;
  }

  template <typename R>
  R sync(std::function<R()> func) {
    return async(std::move(func)).get();
  }

  template <typename F>
  auto sync(F&& func) -> decltype(func()) {
    return async(std::forward<F>(func)).get();
  }

 private:
  GlobalWorkStreamExecutor() {
    worker_ = std::thread([this]() {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(queue_mutex_);
          condition_.wait(lock, [this] { return !tasks_.empty() || stop_; });

          if (stop_ && tasks_.empty()) break;

          task = std::move(tasks_.front());
          tasks_.pop();
        }
        task();  // 执行任务
      }
    });
  }

  ~GlobalWorkStreamExecutor() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  void add_task(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      tasks_.emplace(std::move(task));
    }
    condition_.notify_one();
  }

  // 删除拷贝构造和赋值
  GlobalWorkStreamExecutor(const GlobalWorkStreamExecutor&) = delete;
  GlobalWorkStreamExecutor& operator=(const GlobalWorkStreamExecutor&) = delete;

  std::thread worker_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_ = false;
};
#endif

class RecipeRunner {
 public:
  explicit RecipeRunner(synRecipeHandle h) : recipeHandle_(h) {}
  ~RecipeRunner() {}

  void prepareTensorInfo(synRecipeHandle recipe,
                         synLaunchTensorInfo* tensorInfo,
                         uint32_t totalNumOfTensors);
#ifdef ENABLE_ASYNC_RUN
  void Run(C_Stream stream, std::map<std::string, uint64_t> tensors) {
    synRecipeHandle recipehandle = this->recipeHandle_;
    auto future = GlobalWorkStreamExecutor::instance().async(
        [this, stream, tensors, recipehandle] {
          ExecuteRecipe(stream, tensors, recipehandle);
        });
  }
#else
  void Run(C_Stream stream, const std::map<std::string, uint64_t>& tensors);
#endif

 protected:
#ifdef ENABLE_ASYNC_RUN
  void ExecuteRecipe(C_Stream stream,
                     const std::map<std::string, uint64_t>& tensors,
                     synRecipeHandle recipeHandle_);
#endif

  synRecipeHandle recipeHandle_;

 private:
  C_Status MallocDeviceMem(uint64_t* buffer, const uint64_t size);
  C_Status FreeDeviceMem(const uint64_t buffer, const uint64_t size);
};

#endif  // BACKENDS_INTEL_HPU_KERNELS_HPU_OPERATOR_H_ // NOLINT
