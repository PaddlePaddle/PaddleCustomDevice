// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#pragma once

#include <dlfcn.h>  // dladdr
#include <sdpti.h>  //NOLINT
#include <sys/time.h>

#include <mutex>  // NOLINT

#include "dynload/dynamic_loader.h"

namespace custom_dynload {

extern std::once_flag sdpti_dso_flag;
extern void *sdpti_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load sdpti routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_SDPTI_WRAP(__name)                   \
  struct DynLoad__##__name {                                      \
    template <typename... Args>                                   \
    inline SDptiResult operator()(Args... args) {                 \
      using sdptiFunc = decltype(&::__name);                      \
      std::call_once(sdpti_dso_flag, []() {                       \
        sdpti_dso_handle = custom_dynload::GetSDPTIDsoHandle();   \
      });                                                         \
      if (nullptr == sdpti_dso_handle) {                          \
        if ("sdptiGetVersion" == #__name) {                       \
          return SDPTI_SUCCESS;                                   \
        }                                                         \
        return SDPTI_ERROR_NOT_SUPPORTED;                         \
      }                                                           \
      static void *p_##__name = dlsym(sdpti_dso_handle, #__name); \
      return reinterpret_cast<sdptiFunc>(p_##__name)(args...);    \
    }                                                             \
  };                                                              \
  extern DynLoad__##__name __name

#define SDPTI_ROUTINE_EACH(__macro)        \
  __macro(sdptiGetCallbackName);           \
  __macro(sdptiActivityGetNextRecord);     \
  __macro(sdptiGetVersion);                \
  __macro(sdptiActivityFlushAll);          \
  __macro(sdptiFinalize);                  \
  __macro(sdptiActivitySetAttribute);      \
  __macro(sdptiInit);                      \
  __macro(sdptiActivityRegisterCallbacks); \
  __macro(sdptiActivityEnable);            \
  __macro(sdptiActivityDisable);

SDPTI_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_SDPTI_WRAP);

#undef DECLARE_DYNAMIC_LOAD_SDPTI_WRAP

}  // namespace custom_dynload
