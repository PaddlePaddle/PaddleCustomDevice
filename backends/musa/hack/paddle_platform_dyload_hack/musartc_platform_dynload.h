// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// Modifications:
// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.

// - [modify the code to adapt to musa backend]
#pragma once

#include <mutex>  // NOLINT

#include "musartc_dynload.h"  // NOLINT

namespace paddle {
namespace platform {
namespace dynload {

extern bool HasNVRTC();

#define PLATFORM_DECLARE_DYNAMIC_LOAD_NVRTC_WRAP(__name)     \
  using DynLoad__##__name = phi::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

/**
 * include all needed musartc functions
 **/
#define MUSARTC_ROUTINE_EACH(__macro) \
  __macro(mtrtcVersion);              \
  __macro(mtrtcGetErrorString);       \
  __macro(mtrtcCompileProgram);       \
  __macro(mtrtcCreateProgram);        \
  __macro(mtrtcDestroyProgram);       \
  __macro(mtrtcGetMUSA);              \
  __macro(mtrtcGetMUSASize);          \
  __macro(mtrtcGetProgramLog);        \
  __macro(mtrtcGetProgramLogSize)

MUSARTC_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_NVRTC_WRAP);

#undef PLATFORM_DECLARE_DYNAMIC_LOAD_NVRTC_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
