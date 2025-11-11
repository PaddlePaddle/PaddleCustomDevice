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
#include <string>

#include "common/gcu_env_list.h"
#include "runtime/runtime.h"
#include "topstx/topstx.h"

namespace custom_kernel {
inline bool ProfilerIsOn() {
  static const char *profiler_env = std::getenv(env::kProfiler);
  static bool profiler_is_on =
      (profiler_env != nullptr && std::string(profiler_env) == "true");
  return profiler_is_on;
}

enum TraceCategory {
  PADDLE_GCU_KERNEL,
  GCU_AOT_KERNEL,
  EXEC,
};

class GcuProfiler {
 public:
  static const topstxDomainHandle_t &Domain() {
    static GcuProfiler inst;
    return inst.domain_;
  }

 private:
  GcuProfiler() {
    domain_ = topstxDomainCreate("PADDLE_GCU_BACKEND");
    topstxDomainNameCategory(
        domain_, TraceCategory::PADDLE_GCU_KERNEL, "PADDLE_GCU_KERNEL");
    topstxDomainNameCategory(
        domain_, TraceCategory::GCU_AOT_KERNEL, "GCU_AOT_KERNEL");
    topstxDomainNameCategory(domain_, TraceCategory::EXEC, "EXEC");
  }

  ~GcuProfiler() { topstxDomainDestroy(domain_); }

  topstxDomainHandle_t domain_;
};

class TraceGuardBase {
 public:
  TraceGuardBase(const std::string &op_type, const TraceCategory &category)
      : domain_(GcuProfiler::Domain()) {
    if (UNLIKELY(ProfilerIsOn())) {
      topstxEventAttributes event = {};
      event.size = TOPSTX_EVENT_ATTRIB_STRUCT_SIZE;
      event.messageType = TOPSTX_MESSAGE_TYPE_STRING;
      event.message.str = op_type.c_str();
      event.category = category;
      range_id_ = topstxDomainRangeStart(domain_, &event);
    }
  }

  ~TraceGuardBase() {
    if (UNLIKELY(ProfilerIsOn())) {
      topstxDomainRangeEnd(domain_, range_id_);
    }
  }

  TraceGuardBase() = delete;
  RT_DISALLOW_COPY_AND_ASSIGN(TraceGuardBase);

 private:
  topstxDomainHandle_t domain_;
  topstxRangeId_t range_id_;
};

#define PADDLE_GCU_KERNEL_TRACE(kernel_type)              \
  custom_kernel::TraceGuardBase paddle_gcu_kernel_tracer( \
      (kernel_type), custom_kernel::TraceCategory::PADDLE_GCU_KERNEL)

#define GCU_AOT_KERNEL_TRACE(kernel_type)              \
  custom_kernel::TraceGuardBase gcu_aot_kernel_tracer( \
      (kernel_type), custom_kernel::TraceCategory::GCU_AOT_KERNEL)

}  // namespace custom_kernel
