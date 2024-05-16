/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <unordered_map>

#include "paddle/phi/core/allocator.h"
#include "runtime/runtime.h"

namespace custom_kernel {

class PinnedAllocator : public phi::Allocator {
 public:
  static void Deleter(phi::Allocation* allocation) {
    (void)PinnedDeallocate(allocation->ptr(), allocation->size());
  }

  AllocationPtr Allocate(size_t bytes_size) override {
    void* data_ptr = nullptr;
    auto ret = PinnedAllocate(&data_ptr, bytes_size);
    PADDLE_ENFORCE_EQ(
        ret,
        C_SUCCESS,
        phi::errors::Fatal("Failed to alloc pinned memory with size %zu.",
                           bytes_size));
    auto* allocation =
        new phi::Allocation(data_ptr, bytes_size, phi::CPUPlace());
    return AllocationPtr(allocation, Deleter);
  }
};

class AllocatorManager {
 public:
  ~AllocatorManager() = default;

  static AllocatorManager& Instance() {
    static AllocatorManager instance;
    return instance;
  }

  std::shared_ptr<phi::Allocator> GetAllocator(
      const std::string& name = "Pinned") {
    PADDLE_ENFORCE_NE(
        allocators_.count(name),
        0,
        phi::errors::NotFound("Not found alloctor %s.", name.c_str()));
    return allocators_.at(name);
  }

  RT_DISALLOW_COPY_AND_ASSIGN(AllocatorManager);

 private:
  AllocatorManager() { Init(); }
  void Init() {
    allocators_.emplace("Pinned", std::make_shared<PinnedAllocator>());
  }
  std::unordered_map<std::string, std::shared_ptr<phi::Allocator>> allocators_;
};

template <typename Context>
class ContextPinnedGuard {
 public:
  explicit ContextPinnedGuard(const Context& dev_ctx)
      : ctx_(const_cast<Context*>(&dev_ctx)),
        allocator_(const_cast<phi::Allocator*>(&(dev_ctx.GetHostAllocator()))) {
    ctx_->SetHostAllocator(
        AllocatorManager::Instance().GetAllocator("Pinned").get());
  }

  ~ContextPinnedGuard() { ctx_->SetHostAllocator(allocator_); }

  ContextPinnedGuard() = delete;
  RT_DISALLOW_COPY_AND_ASSIGN(ContextPinnedGuard);

 private:
  Context* ctx_;
  phi::Allocator* allocator_ = nullptr;
};

template <typename T>
class PinnedAllocatorForSTL : public std::allocator<T> {
 public:
  template <typename U>
  struct rebind {
    typedef PinnedAllocatorForSTL<U> other;
  };

 public:
  T* allocate(size_t n, const void* hint = 0) {
    void* ptr = nullptr;
    auto ret = PinnedAllocate(&ptr, (n * sizeof(T)));
    PADDLE_ENFORCE_EQ(
        ret,
        C_SUCCESS,
        phi::errors::Fatal("Failed to alloc pinned memory with size %zu.",
                           (n * sizeof(T))));
    return static_cast<T*>(ptr);
  }

  void deallocate(T* ptr, size_t n) {
    (void)PinnedDeallocate(ptr, (n * sizeof(T)));
  }
};

}  // namespace custom_kernel
